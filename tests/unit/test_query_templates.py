"""
Unit tests for the construction of SPARQL queries from the query templates.
"""

import sys

from pathlib import Path

import pytest

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vantage6_strongaya_rdf.collect_sparql_data import (  # noqa: E402
    DEFAULT_ONTOLOGY_PREFIXES,
    STRUCTURAL_PREFIXES,
    _build_prefix_declarations,
    _load_query_template,
    _prepare_query_template,
)
from vantage6_strongaya_rdf.schema_loader import load_schema  # noqa: E402
from vantage6_strongaya_rdf.schema_parser import (  # noqa: E402
    DEFAULT_IDENTIFIER_CLASS,
    DEFAULT_IDENTIFIER_PREDICATE,
    get_identifier_query_params,
    get_schema_prefixes,
    get_variable_query_params,
)

QUERY_TEMPLATES = ["single_column", "multi_column"]


def complete_query(query_name: str, variables: list, schema: dict) -> str:
    """
    Complete a query template just like the data collection functions do.

    :param query_name: Name of the query template
    :param variables: Variables (or class codes) to query
    :param schema: The full schema dictionary
    :return: The completed SPARQL query
    """
    query = _prepare_query_template(_load_query_template(query_name), schema)

    if query_name == "single_column":
        params = get_variable_query_params(variables[0], schema)
        return (
            query.replace("PLACEHOLDER_CLASS", params["main_class"])
            .replace("PLACEHOLDER_ONTOLOGY", params["ontology_prefix"])
            .replace("PLACEHOLDER_PREDICATE_PATH", params["predicate_path"])
        )

    for index, variable in enumerate(variables, start=1):
        params = get_variable_query_params(variable, schema)
        query = (
            query.replace(f"PLACEHOLDER_CLASS_{index}", params["main_class"])
            .replace(f"PLACEHOLDER_ONTOLOGY_{index}", params["ontology_prefix"])
            .replace(f"PLACEHOLDER_PREDICATE_PATH_{index}", params["predicate_path"])
        )
    return query


class TestQueryTemplates:
    """Test the query templates themselves."""

    @pytest.mark.parametrize("query_name", QUERY_TEMPLATES)
    def test_template_declares_no_ontology_prefixes(self, query_name):
        """Test that the templates hold no hardcoded ontology prefixes.

        The ontologies that the schema describes would otherwise be able to diverge
        from the queries that are built for that very schema.
        """
        template = _load_query_template(query_name)

        assert "PLACEHOLDER_PREFIXES" in template
        for prefix in ["ncit", "sio", "sct", "strongaya", "roo"]:
            assert (
                f"PREFIX {prefix}:" not in template
            ), f"Template '{query_name}' still declares the '{prefix}' prefix"

    @pytest.mark.parametrize("query_name", QUERY_TEMPLATES)
    def test_template_derives_the_identifier(self, query_name):
        """Test that the templates hold no hardcoded identifier predicate or class."""
        template = _load_query_template(query_name)

        assert "PLACEHOLDER_ID_PREDICATE" in template
        assert "PLACEHOLDER_ID_CLASS" in template
        assert DEFAULT_IDENTIFIER_PREDICATE not in template
        assert DEFAULT_IDENTIFIER_CLASS not in template


class TestPrefixDeclarations:
    """Test the composition of the PREFIX declarations."""

    @classmethod
    def setup_class(cls):
        """Load schema once for all tests."""
        cls.schema = load_schema(use_remote=False)

    def test_prefixes_are_derived_from_the_schema(self):
        """Test that every prefix of the schema is declared."""
        declarations = _build_prefix_declarations(self.schema)

        for prefix, uri in get_schema_prefixes(self.schema).items():
            if prefix in STRUCTURAL_PREFIXES:
                continue
            assert (
                f"PREFIX {prefix}: <{uri}>" in declarations
            ), f"Prefix '{prefix}' of the schema is not declared"

    def test_structural_prefixes_are_always_declared(self):
        """Test that the Triplifier's prefixes are declared, as the schema lacks them."""
        for schema in [None, self.schema]:
            declarations = _build_prefix_declarations(schema)

            for prefix, uri in STRUCTURAL_PREFIXES.items():
                assert f"PREFIX {prefix}: <{uri}>" in declarations

    def test_schema_prefixes_take_precedence_over_the_defaults(self):
        """Test that the schema's own namespaces are used.

        The templates used to declare sct: as <http://snomed.info/id/>, whereas the
        schema declares <http://purl.bioontology.org/ontology/SNOMEDCT/>; such a
        divergence silently yields queries that cannot match any data.
        """
        schema_prefixes = get_schema_prefixes(self.schema)
        declarations = _build_prefix_declarations(self.schema)

        divergent_prefixes = [
            prefix
            for prefix, uri in DEFAULT_ONTOLOGY_PREFIXES.items()
            if prefix in schema_prefixes and schema_prefixes[prefix] != uri
        ]
        assert divergent_prefixes, "Expected at least one divergent default prefix"

        for prefix in divergent_prefixes:
            assert f"PREFIX {prefix}: <{schema_prefixes[prefix]}>" in declarations
            assert f"PREFIX {prefix}: <{DEFAULT_ONTOLOGY_PREFIXES[prefix]}>" not in (
                declarations
            )

    def test_default_prefixes_are_used_without_a_schema(self):
        """Test that queries can still be built when no schema is used."""
        declarations = _build_prefix_declarations(None)

        for prefix, uri in DEFAULT_ONTOLOGY_PREFIXES.items():
            assert f"PREFIX {prefix}: <{uri}>" in declarations

    def test_declarations_are_deterministic(self):
        """Test that the declarations do not depend on the schema's key order."""
        assert _build_prefix_declarations(self.schema) == _build_prefix_declarations(
            self.schema
        )


class TestIdentifierQueryParams:
    """Test deriving the patient's identifier from the schema."""

    @classmethod
    def setup_class(cls):
        """Load schema once for all tests."""
        cls.schema = load_schema(use_remote=False)

    def test_identifier_is_derived_from_the_schema(self):
        """Test that the identifier's predicate and class come from the schema."""
        params = get_identifier_query_params(self.schema)

        assert params == {
            "predicate": DEFAULT_IDENTIFIER_PREDICATE,
            "class": DEFAULT_IDENTIFIER_CLASS,
        }

    def test_identifier_predicate_is_a_single_predicate(self):
        """Test that the identifier is not queried through a transitive path.

        A transitive path such as "(sio:SIO_000673)*" also matches the patient node
        itself through its zero-length match, which yields spurious identifiers.
        """
        params = get_identifier_query_params(self.schema)

        assert "*" not in params["predicate"]
        assert "|" not in params["predicate"]
        assert "/" not in params["predicate"]

    def test_identifier_of_a_custom_schema(self):
        """Test that a schema's own identifier definition is honoured."""
        params = get_identifier_query_params(
            {
                "schema": {
                    "variables": {
                        "patient_number": {
                            "@type": "schema:IdentifierVariable",
                            "dataType": "identifier",
                            "predicate": "sio:SIO_000123",
                            "class": "ncit:C12345",
                        }
                    }
                }
            }
        )

        assert params == {"predicate": "sio:SIO_000123", "class": "ncit:C12345"}

    def test_identifier_falls_back_without_a_definition(self):
        """Test that a schema without an identifier does not break query construction."""
        params = get_identifier_query_params({})

        assert params == {
            "predicate": DEFAULT_IDENTIFIER_PREDICATE,
            "class": DEFAULT_IDENTIFIER_CLASS,
        }


class TestCompletedQueries:
    """Test the queries that are handed to the SPARQL endpoint."""

    @classmethod
    def setup_class(cls):
        """Load schema once for all tests."""
        cls.schema = load_schema(use_remote=False)

    @pytest.mark.parametrize(
        "query_name, variables",
        [
            ("single_column", ["ncit:C28421"]),
            ("multi_column", ["ncit:C28421", "ncit:C156420"]),
        ],
    )
    def test_completed_query_holds_no_placeholders(self, query_name, variables):
        """Test that every placeholder of a template is substituted."""
        query = complete_query(query_name, variables, self.schema)

        assert "PLACEHOLDER" not in query

    @pytest.mark.parametrize(
        "query_name, variables",
        [
            ("single_column", ["ncit:C28421"]),
            ("single_column", ["ncit:C192402"]),
            ("multi_column", ["ncit:C28421", "ncit:C156420"]),
        ],
    )
    def test_completed_query_is_valid_sparql(self, query_name, variables):
        """Test that the completed queries are syntactically valid SPARQL."""
        rdflib_plugins = pytest.importorskip(
            "rdflib.plugins.sparql", reason="rdflib is required to parse SPARQL"
        )

        query = complete_query(query_name, variables, self.schema)

        # Raises when the query - including its property paths - cannot be parsed
        rdflib_plugins.prepareQuery(query)

    @pytest.mark.parametrize("query_name", QUERY_TEMPLATES)
    def test_completed_query_declares_the_prefixes_it_uses(self, query_name):
        """Test that the prefixes that the query body uses are declared."""
        variables = (
            ["ncit:C28421"]
            if query_name == "single_column"
            else ["ncit:C28421", "ncit:C156420"]
        )
        query = complete_query(query_name, variables, self.schema)

        declarations, _, body = query.partition("SELECT")
        for prefix in ["dbo", "rdf", "rdfs", "ncit", "sio", "strongaya"]:
            assert f"{prefix}:" in body, f"Query does not use the '{prefix}' prefix"
            assert (
                f"PREFIX {prefix}:" in declarations
            ), f"Query does not declare the '{prefix}' prefix"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
