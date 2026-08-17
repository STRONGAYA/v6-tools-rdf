"""
Unit tests that hold an entire schema to the contract that the queries rely on.

The remainder of the suite pins the predicate path of a handful of variables, whereas
the semantic map describes hundreds of them and is bumped automatically by the
sync-schema workflow. The tests below therefore verify every variable of the schema
at once, so that a release which breaks the contract for a single variable - a path
that cannot be built, a path that turns transitive, or an ontology prefix that is not
declared - is noticed before a node runs against it.

Three groups of tests are collected here:

1. The invariants of every variable of the bundled schema;
2. A golden snapshot of the predicate paths, which turns a schema update into a
   reviewable diff rather than a silent change; and
3. An opt-in test that applies the very same invariants to the latest upstream
   release, so that a broken contract is learned about upstream rather than at a node.
"""

import json
import os
import re
import sys

from pathlib import Path
from typing import Dict, List, Set

import pytest

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vantage6_strongaya_rdf.collect_sparql_data import (  # noqa: E402
    _build_prefix_declarations,
    _load_query_template,
    _prepare_query_template,
)
from vantage6_strongaya_rdf.schema_loader import (  # noqa: E402
    BUNDLED_SCHEMA_TAG,
    build_schema_url,
    get_schema_version,
    load_schema,
    resolve_latest_schema_tag,
    validate_schema,
)
from vantage6_strongaya_rdf.schema_parser import (  # noqa: E402
    build_predicate_path,
    get_variable_query_params,
)

sparql_plugins = pytest.importorskip(
    "rdflib.plugins.sparql", reason="rdflib is required to parse SPARQL"
)

# Location and format of the golden snapshot of the predicate paths
SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "predicate_paths.json"
SNAPSHOT_ENVIRONMENT_VARIABLE = "UPDATE_PREDICATE_PATH_SNAPSHOT"
SNAPSHOT_COMMAND = (
    f"{SNAPSHOT_ENVIRONMENT_VARIABLE}=1 "
    "python -m pytest tests/unit/test_schema_contract.py"
)

# Environment variable that opts in to the tests that require the internet
NETWORK_ENVIRONMENT_VARIABLE = "RUN_NETWORK_TESTS"

# Path operators that would match the patient node itself or an unrelated branch
FORBIDDEN_PATH_OPERATORS = ["*", "?", "+"]

# The prefix of a prefixed name, such as the "sio" of "sio:SIO_000255"
PREFIXED_NAME_PATTERN = re.compile(r"([A-Za-z][\w.\-]*):")


def _schema_variables(schema: dict) -> dict:
    """
    Retrieve the variable definitions of a schema.

    :param schema: The full schema dictionary
    :return: Dictionary of variable definitions
    """
    return schema.get("schema", {}).get("variables", {}) or {}


def _query_params(schema: dict) -> Dict[str, dict]:
    """
    Build the query parameters of every variable of a schema.

    :param schema: The full schema dictionary
    :return: Dictionary that maps each variable name to its query parameters
    """
    return {
        name: get_variable_query_params(name, schema)
        for name in _schema_variables(schema)
    }


def _declared_prefixes(schema: dict) -> Set[str]:
    """
    Determine the prefixes that a query built for a schema declares.

    :param schema: The full schema dictionary
    :return: Set of declared prefixes, without their colon
    """
    declarations = _build_prefix_declarations(schema)
    return set(re.findall(r"PREFIX (\S+):", declarations))


def _used_prefixes(*prefixed_names: str) -> Set[str]:
    """
    Determine the prefixes that prefixed names use.

    :param prefixed_names: Prefixed names, such as a predicate path or a class
    :return: Set of used prefixes, without their colon
    """
    return {
        prefix
        for prefixed_name in prefixed_names
        for prefix in PREFIXED_NAME_PATTERN.findall(prefixed_name or "")
    }


def _complete_single_column_query(prepared_template: str, params: dict) -> str:
    """
    Complete the single-column template just like the data collection functions do.

    :param prepared_template: The template with its prefixes and identifier filled in
    :param params: The query parameters of the variable to query
    :return: The completed SPARQL query
    """
    return (
        prepared_template.replace("PLACEHOLDER_CLASS", params.get("main_class", ""))
        .replace("PLACEHOLDER_ONTOLOGY", params.get("ontology_prefix", ""))
        .replace("PLACEHOLDER_PREDICATE_PATH", params.get("predicate_path", ""))
    )


def _variables_without_a_path(schema: dict) -> List[str]:
    """
    Determine the variables of a schema for which no predicate path can be built.

    :param schema: The full schema dictionary
    :return: Report of every offending variable
    """
    return [
        name
        for name in sorted(_schema_variables(schema))
        if not build_predicate_path(name, schema)
    ]


def _variables_with_a_forbidden_operator(params: Dict[str, dict]) -> List[str]:
    """
    Determine the variables whose predicate path holds a transitive or optional operator.

    :param params: The query parameters of every variable
    :return: Report of every offending variable
    """
    offenders = []
    for name, variable_params in sorted(params.items()):
        path = variable_params.get("predicate_path", "")
        if any(operator in path for operator in FORBIDDEN_PATH_OPERATORS):
            offenders.append(f"{name}: {path}")
    return offenders


def _variables_with_an_undeclared_prefix(
    params: Dict[str, dict], declared: Set[str]
) -> List[str]:
    """
    Determine the variables that use a prefix which no query declares.

    :param params: The query parameters of every variable
    :param declared: The prefixes that a query built for the schema declares
    :return: Report of every offending variable
    """
    offenders = []
    for name, variable_params in sorted(params.items()):
        undeclared = sorted(
            _used_prefixes(
                variable_params.get("predicate_path", ""),
                variable_params.get("main_class", ""),
            )
            - declared
        )
        if undeclared:
            offenders.append(f"{name}: {', '.join(undeclared)}")
    return offenders


def _variables_with_an_unparsable_query(
    prepared_template: str, params: Dict[str, dict]
) -> List[str]:
    """
    Determine the variables whose completed query cannot be parsed as SPARQL.

    :param prepared_template: The template with its prefixes and identifier filled in
    :param params: The query parameters of every variable
    :return: Report of every offending variable
    """
    offenders = []
    for name, variable_params in sorted(params.items()):
        query = _complete_single_column_query(prepared_template, variable_params)
        try:
            sparql_plugins.prepareQuery(query)
        except Exception as e:
            offenders.append(f"{name}: {e}")
    return offenders


def _contract_failures(schema: dict) -> List[str]:
    """
    Verify every invariant of every variable of a schema.

    :param schema: The full schema dictionary
    :return: Report of every violated invariant
    """
    params = _query_params(schema)
    prepared_template = _prepare_query_template(
        _load_query_template("single_column"), schema
    )

    failures = []
    for description, offenders in [
        ("no predicate path could be built", _variables_without_a_path(schema)),
        (
            "the predicate path is transitive or optional",
            _variables_with_a_forbidden_operator(params),
        ),
        (
            "a prefix is not declared by the query",
            _variables_with_an_undeclared_prefix(params, _declared_prefixes(schema)),
        ),
        (
            "the completed query is not valid SPARQL",
            _variables_with_an_unparsable_query(prepared_template, params),
        ),
    ]:
        failures.extend(f"{description} - {offender}" for offender in offenders)
    return failures


def _build_snapshot(schema: dict) -> dict:
    """
    Compose the golden snapshot of the predicate paths of a schema.

    :param schema: The full schema dictionary
    :return: The snapshot of the schema's release and of each of its variables
    """
    return {
        "schema_tag": BUNDLED_SCHEMA_TAG,
        "schema_version": get_schema_version(schema),
        "variables": {
            name: {
                "class": variable_params.get("main_class", ""),
                "ontology_prefix": variable_params.get("ontology_prefix", ""),
                "predicate_path": variable_params.get("predicate_path", ""),
            }
            for name, variable_params in _query_params(schema).items()
        },
    }


def _serialise_snapshot(snapshot: dict) -> str:
    """
    Serialise a snapshot the way it is stored, so that its diff stays reviewable.

    :param snapshot: The snapshot to serialise
    :return: The snapshot as sorted, indented JSON with a trailing newline
    """
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def _write_snapshot(snapshot: dict) -> None:
    """
    Store a snapshot on disk.

    :param snapshot: The snapshot to store
    """
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(_serialise_snapshot(snapshot), encoding="utf-8")


def _read_snapshot() -> dict:
    """
    Retrieve the snapshot that is stored on disk.

    :return: The recorded snapshot
    """
    assert SNAPSHOT_PATH.exists(), (
        f"The predicate path snapshot is missing at {SNAPSHOT_PATH}; "
        f"generate it with '{SNAPSHOT_COMMAND}'"
    )
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _describe_differences(recorded: dict, regenerated: dict) -> List[str]:
    """
    Describe how the regenerated predicate paths differ from the recorded ones.

    :param recorded: The variables of the snapshot that is stored on disk
    :param regenerated: The variables as derived from the schema
    :return: Description of every difference between the two
    """
    differences = []

    for name in sorted(set(recorded) - set(regenerated)):
        differences.append(f"- {name}: no longer described by the schema")

    for name in sorted(set(regenerated) - set(recorded)):
        differences.append(f"+ {name}: {regenerated[name]['predicate_path']}")

    for name in sorted(set(recorded) & set(regenerated)):
        if recorded[name] != regenerated[name]:
            differences.append(f"~ {name}: {recorded[name]} became {regenerated[name]}")

    return differences


class TestWholeSchemaInvariants:
    """Test the invariants that hold for every variable of the bundled schema.

    Each test collects all of its offenders before it fails, so that a schema update
    is reviewed as a whole rather than one variable at a time.

    Parsing the completed query of every variable is the most expensive invariant;
    it takes some eight seconds for the schema's variables, which is well within the
    budget of the unit suite. Should that ever exceed roughly thirty seconds, the
    queries can be parsed per distinct predicate path instead - the paths of the
    schema collapse into far fewer distinct paths than there are variables - at the
    cost of no longer naming every affected variable.
    """

    @classmethod
    def setup_class(cls):
        """Load schema once for all tests."""
        cls.schema = load_schema(use_remote=False)
        cls.variables = _schema_variables(cls.schema)
        cls.params = _query_params(cls.schema)
        cls.prepared_template = _prepare_query_template(
            _load_query_template("single_column"), cls.schema
        )

    def test_the_schema_describes_its_variables(self):
        """Test that the bundled schema holds the variables that the tests verify.

        The invariants below would pass vacuously were the schema to hold no
        variables at all, for instance after a failed schema update.
        """
        assert len(self.variables) > 100, (
            f"The bundled schema describes only {len(self.variables)} variables; "
            "the invariants below would hardly verify anything"
        )

    def test_every_variable_yields_a_predicate_path(self):
        """Test that a predicate path can be built for every variable.

        A variable without a path is queried through no structure at all, which
        yields either no data or the data of an unrelated branch.
        """
        offenders = _variables_without_a_path(self.schema)

        assert not offenders, (
            f"{len(offenders)} variable(s) of schema version "
            f"{get_schema_version(self.schema)} yield no predicate path:\n"
            + "\n".join(offenders)
        )

    def test_no_predicate_path_is_transitive_or_optional(self):
        """Test that no predicate path holds a transitive or optional operator.

        A path such as "(a|b)*" matches the predicates in any order, at any depth
        and - through its zero-length match - the patient node itself, which allows
        the values of unrelated branches to be collected as though they were the
        variable's own.
        """
        offenders = _variables_with_a_forbidden_operator(self.params)

        assert not offenders, (
            f"{len(offenders)} variable(s) are queried through a transitive or "
            "optional path:\n" + "\n".join(offenders)
        )

    def test_every_prefix_of_every_variable_is_declared(self):
        """Test that the prefixes of every path and class are declared by the query.

        A prefix that is used but not declared renders the query unusable, and the
        schema is free to introduce an ontology that the prefix extraction does not
        yet know about.
        """
        offenders = _variables_with_an_undeclared_prefix(
            self.params, _declared_prefixes(self.schema)
        )

        assert not offenders, (
            f"{len(offenders)} variable(s) use a prefix that the query does not "
            "declare:\n" + "\n".join(offenders)
        )

    def test_every_variable_yields_a_valid_query(self):
        """Test that the completed query of every variable is valid SPARQL.

        The prefixes of a query are derived from the schema, which means that a
        schema that introduces an ontology whose prefix is not picked up yields an
        unparsable query for the variables of that ontology alone; such a break
        would remain unnoticed by the handful of variables that are pinned
        elsewhere in the suite.
        """
        offenders = _variables_with_an_unparsable_query(
            self.prepared_template, self.params
        )

        assert not offenders, (
            f"{len(offenders)} variable(s) yield a query that cannot be parsed:\n"
            + "\n".join(offenders)
        )


class TestPredicatePathSnapshot:
    """Test the golden snapshot of the predicate paths of the bundled schema."""

    @classmethod
    def setup_class(cls):
        """Load schema once for all tests."""
        cls.schema = load_schema(use_remote=False)
        cls.snapshot = _build_snapshot(cls.schema)

    def test_snapshot_matches_the_bundled_schema(self):
        """Test that the recorded predicate paths are the ones that are built today.

        The schema is bumped automatically, which means that the structure through
        which a variable is queried can change without anybody noticing. Recording
        the paths turns such a change into a reviewable diff of the pull request
        that bumps the schema.
        """
        if os.environ.get(SNAPSHOT_ENVIRONMENT_VARIABLE):
            _write_snapshot(self.snapshot)
            print(f"Updated the predicate path snapshot at {SNAPSHOT_PATH}")

        recorded = _read_snapshot()
        differences = _describe_differences(
            recorded.get("variables", {}), self.snapshot["variables"]
        )

        assert not differences, (
            f"The predicate paths of schema version "
            f"{get_schema_version(self.schema)} differ from the recorded ones "
            f"({len(differences)} difference(s)):\n"
            + "\n".join(differences)
            + "\n\nReview whether the schema intends these paths and, when it does, "
            f"regenerate the snapshot with '{SNAPSHOT_COMMAND}'"
        )

    def test_snapshot_records_the_bundled_schema_release(self):
        """Test that the snapshot belongs to the schema that is bundled.

        A snapshot of another release would keep the comparison above green whilst
        describing paths that no node ever queries.
        """
        recorded = _read_snapshot()

        assert recorded.get("schema_version") == get_schema_version(self.schema), (
            f"The snapshot records schema version {recorded.get('schema_version')} "
            f"whereas the bundled schema is version "
            f"{get_schema_version(self.schema)}; regenerate the snapshot with "
            f"'{SNAPSHOT_COMMAND}'"
        )
        assert recorded.get("schema_tag") == BUNDLED_SCHEMA_TAG, (
            f"The snapshot records release {recorded.get('schema_tag')} whereas the "
            f"library bundles {BUNDLED_SCHEMA_TAG}; regenerate the snapshot with "
            f"'{SNAPSHOT_COMMAND}'"
        )

    def test_snapshot_is_stored_as_it_is_generated(self):
        """Test that the stored snapshot is formatted the way it is regenerated.

        A snapshot that is edited by hand yields a diff full of reordered or
        reindented lines, which is precisely what the snapshot is meant to avoid.
        """
        assert SNAPSHOT_PATH.read_text(encoding="utf-8") == _serialise_snapshot(
            self.snapshot
        ), (
            "The stored snapshot is not the verbatim result of a regeneration; "
            f"regenerate it with '{SNAPSHOT_COMMAND}'"
        )


class TestUpstreamSchemaContract:
    """Test the contract against the schema that upstream publishes."""

    @pytest.mark.network
    @pytest.mark.skipif(
        not os.environ.get(NETWORK_ENVIRONMENT_VARIABLE),
        reason=(
            f"Set {NETWORK_ENVIRONMENT_VARIABLE} to verify the latest upstream "
            "release of the semantic map"
        ),
    )
    def test_latest_upstream_schema_honours_the_contract(self):
        """Test that the latest published schema still honours the query contract.

        The schema is bumped automatically, so a release that no longer satisfies
        the invariants of the bundled schema would first be noticed by a node that
        collects no data at all. Fetching the release here means that such a break
        is learned about upstream instead. The test is opt-in, as the default suite
        must not depend on the internet.
        """
        schema_tag = resolve_latest_schema_tag()
        if not schema_tag:
            pytest.skip("The latest upstream release could not be resolved")

        try:
            schema = load_schema(
                use_remote=True, schema_tag=schema_tag, local_fallback=False
            )
        except Exception as e:
            pytest.skip(
                f"The schema of release {schema_tag} could not be fetched from "
                f"{build_schema_url(schema_tag)}: {e}"
            )

        # Raises when the document is not a semantic map of the expected shape
        validate_schema(schema)

        failures = _contract_failures(schema)

        assert not failures, (
            f"Upstream release {schema_tag} (schema version "
            f"{get_schema_version(schema)}) breaks the query contract in "
            f"{len(failures)} case(s):\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
