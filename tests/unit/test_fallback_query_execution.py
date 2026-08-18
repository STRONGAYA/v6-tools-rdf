"""
Unit tests that execute the generated SPARQL queries in their fallback mode.

The fallback mode (use_schema=False) is the documented default of collect_sparql_data
and reaches a variable through a single predicate - dbo:has_column unless another
property is given - rather than through the structure that the schema describes. It is
the mode that a node runs when no semantic map is available, which is why it is
verified against a graph rather than only through its query template.

The graph mirrors a Triplifier-produced graph that carries an annotation, in which the
value nodes are reachable in both manners:

    record ─sio:SIO_000673─> identifier node ─dbo:has_cell/dbo:has_value─> "ID_0001"
    record ─sio:SIO_000255─> characteristics ─sio:SIO_000008─> variable node
    record ─dbo:has_column──────────────────────────────────> variable node
    variable node ─dbo:has_cell─> cell ─dbo:has_value─> value

Because both routes reach the very same nodes, the fallback mode and the schema mode
are expected to observe exactly the same data; any difference between the two is a
difference in the queries rather than in the data.

Three records are described, each of a single flat table that holds a categorical
variable (the biological sex, whose value is a subclass) and a continuous one (the age
at initial diagnosis). Their number is kept small deliberately, as rdflib evaluates the
multi-column query considerably slower than an RDF-store does.

Type inference is not emulated; the classes that a store would infer through
owl:equivalentClass and rdfs:subClassOf are asserted directly.
"""

import sys

from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# The package re-exports the collect_sparql_data function under the name of its module,
# which is why the module itself is imported explicitly
data_collection = import_module("vantage6_strongaya_rdf.collect_sparql_data")

rdflib = pytest.importorskip(
    "rdflib", reason="rdflib is required to execute the generated SPARQL queries"
)

# Classes of the AYA cancer schema that the synthetic graph describes
BIOLOGICAL_SEX = "ncit:C28421"
AGE_AT_INITIAL_DIAGNOSIS = "ncit:C156420"
MALE = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C20197"
FEMALE = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C16576"

# Predicate of an annotation that the synthetic graph does not describe
UNUSED_VARIABLE_PROPERTY = "dbo:has_unannotated_column"

# Environment variables that would otherwise override the arguments under test
OVERRIDING_ENVIRONMENT_VARIABLES = [
    "SPARQL_ENDPOINT",
    "VARIABLE_PROPERTY",
    "MISSING_DATA_NOTATION",
    "USE_REMOTE_SCHEMA",
    "SCHEMA_URL",
    "SCHEMA_TAG",
]

SYNTHETIC_GRAPH = """
@prefix dbo: <http://um-cds/ontologies/databaseontology/> .
@prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sio: <http://semanticscience.org/resource/> .
@prefix data: <http://data.local/rdf/data/> .

# ---------- Subclasses that a store would infer ----------
ncit:C20197 rdfs:subClassOf ncit:C28421 .
ncit:C16576 rdfs:subClassOf ncit:C28421 .

# ---------- ID_0001 ----------
data:record_one sio:SIO_000673 data:record_one_id ;
                dbo:has_column data:record_one_id .
data:record_one_id a ncit:C25364 ;
                   dbo:has_cell data:record_one_id_cell .
data:record_one_id_cell dbo:has_value "ID_0001" .

data:record_one sio:SIO_000255 data:record_one_characteristics ;
                dbo:has_column data:record_one_sex ,
                               data:record_one_age .
data:record_one_characteristics sio:SIO_000008 data:record_one_sex ,
                                               data:record_one_age .
data:record_one_sex a ncit:C28421 ;
                    dbo:has_cell data:record_one_sex_cell .
data:record_one_sex_cell a ncit:C20197 ;
                         dbo:has_value "male" .
data:record_one_age a ncit:C156420 ;
                    dbo:has_cell data:record_one_age_cell .
data:record_one_age_cell dbo:has_value "27" .

# ---------- ID_0002 ----------
data:record_two sio:SIO_000673 data:record_two_id ;
                dbo:has_column data:record_two_id .
data:record_two_id a ncit:C25364 ;
                   dbo:has_cell data:record_two_id_cell .
data:record_two_id_cell dbo:has_value "ID_0002" .

data:record_two sio:SIO_000255 data:record_two_characteristics ;
                dbo:has_column data:record_two_sex ,
                               data:record_two_age .
data:record_two_characteristics sio:SIO_000008 data:record_two_sex ,
                                               data:record_two_age .
data:record_two_sex a ncit:C28421 ;
                    dbo:has_cell data:record_two_sex_cell .
data:record_two_sex_cell a ncit:C16576 ;
                         dbo:has_value "female" .
data:record_two_age a ncit:C156420 ;
                    dbo:has_cell data:record_two_age_cell .
data:record_two_age_cell dbo:has_value "31" .

# ---------- ID_0003 ----------
data:record_three sio:SIO_000673 data:record_three_id ;
                  dbo:has_column data:record_three_id .
data:record_three_id a ncit:C25364 ;
                     dbo:has_cell data:record_three_id_cell .
data:record_three_id_cell dbo:has_value "ID_0003" .

data:record_three sio:SIO_000255 data:record_three_characteristics ;
                  dbo:has_column data:record_three_sex ,
                                 data:record_three_age .
data:record_three_characteristics sio:SIO_000008 data:record_three_sex ,
                                                 data:record_three_age .
data:record_three_sex a ncit:C28421 ;
                      dbo:has_cell data:record_three_sex_cell .
data:record_three_sex_cell a ncit:C16576 ;
                           dbo:has_value "female" .
data:record_three_age a ncit:C156420 ;
                      dbo:has_cell data:record_three_age_cell .
data:record_three_age_cell dbo:has_value "19" .
"""


@pytest.fixture(scope="module")
def graph():
    """Provide the synthetic RDF graph."""
    synthetic_graph = rdflib.Graph()
    synthetic_graph.parse(data=SYNTHETIC_GRAPH, format="turtle")
    return synthetic_graph


@pytest.fixture
def sparql_endpoint(monkeypatch, graph):
    """Execute the queries that the library posts against the synthetic graph.

    The results are returned the way that an RDF-store returns them; as strings, with
    an empty string for the variables that a solution leaves unbound. The environment
    variables that the library prioritises over its arguments are removed, as a node's
    configuration must not decide which mode these tests exercise.
    """
    for variable_name in OVERRIDING_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)

    posted_queries = []

    def execute_query(endpoint: str, query: str, **kwargs):
        posted_queries.append(query)
        result = graph.query(query)
        return [
            {
                str(variable): ("" if value is None else str(value))
                for variable, value in zip(result.vars, row)
            }
            for row in result
        ]

    monkeypatch.setattr(data_collection, "post_sparql_query", execute_query)
    return posted_queries


def collect(
    variables,
    query_type="single_column",
    use_schema=False,
    variable_property=None,
):
    """
    Collect data through the library, as an algorithm would.

    :param variables: Variables (or class codes) to collect
    :param query_type: The type of query to use
    :param use_schema: Whether to derive the predicate paths from the schema
    :param variable_property: The predicate that reaches a variable in fallback mode
    :return: pd.DataFrame with the collected data
    """
    return data_collection.collect_sparql_data(
        variables,
        query_type=query_type,
        endpoint="http://localhost:7200/repositories/synthetic",
        variable_property=variable_property,
        use_schema=use_schema,
    )


class TestFallbackSingleColumnQuery:
    """Test the single-column query in its fallback mode."""

    def test_categorical_variable(self, sparql_endpoint):
        """Test that a categorical variable is collected as its subclass.

        The fallback mode is the documented default of the library, which means that
        an algorithm that does not opt into the schema relies on this route entirely.
        """
        result = collect([BIOLOGICAL_SEX])

        values = dict(zip(result["patient_id"], result[BIOLOGICAL_SEX]))
        assert values == {"ID_0001": MALE, "ID_0002": FEMALE, "ID_0003": FEMALE}

    def test_continuous_variable(self, sparql_endpoint):
        """Test that a continuous variable is collected as its value.

        A continuous variable holds no subclass, so its value is taken from the cell
        rather than from the class of that cell.
        """
        result = collect([AGE_AT_INITIAL_DIAGNOSIS])

        values = dict(zip(result["patient_id"], result[AGE_AT_INITIAL_DIAGNOSIS]))
        assert values == {"ID_0001": "27", "ID_0002": "31", "ID_0003": "19"}

    def test_fallback_predicate_is_the_default(self, sparql_endpoint):
        """Test that the fallback mode reaches a variable through dbo:has_column.

        The predicate is documented as the default of the library; a query that used
        the schema's structure instead would find no data in an unannotated graph.
        """
        collect([BIOLOGICAL_SEX])

        posted_query = sparql_endpoint[0]
        assert "?patient dbo:has_column ?sub_class_type" in posted_query
        # The structure that the schema describes is not part of a fallback query
        assert "sio:SIO_000008" not in posted_query

    def test_variables_are_merged_on_the_patient(self, sparql_endpoint):
        """Test that separately queried variables are merged on the patient.

        Each variable is fetched through a query of its own, so their results are only
        of use once they are lined up on the patient that they describe.
        """
        result = collect([BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS]).set_index(
            "patient_id"
        )

        assert result.loc["ID_0001", BIOLOGICAL_SEX] == MALE
        assert result.loc["ID_0001", AGE_AT_INITIAL_DIAGNOSIS] == "27"
        assert result.loc["ID_0003", BIOLOGICAL_SEX] == FEMALE
        assert result.loc["ID_0003", AGE_AT_INITIAL_DIAGNOSIS] == "19"


class TestFallbackMultiColumnQuery:
    """Test the multi-column query in its fallback mode."""

    def test_categorical_and_continuous_variable(self, sparql_endpoint):
        """Test that both variables are fetched together in the fallback mode.

        The multi-column query fills in the fallback predicate twice, once for either
        variable; a mode that is only exercised through the single-column query would
        leave the second of those substitutions unverified.
        """
        result = collect(
            [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS], query_type="multi_column"
        ).set_index("patient_id")

        assert set(result.index) == {"ID_0001", "ID_0002", "ID_0003"}
        assert result.loc["ID_0001", BIOLOGICAL_SEX] == MALE
        assert result.loc["ID_0001", AGE_AT_INITIAL_DIAGNOSIS] == "27"
        assert result.loc["ID_0002", BIOLOGICAL_SEX] == FEMALE
        assert result.loc["ID_0002", AGE_AT_INITIAL_DIAGNOSIS] == "31"
        assert result.loc["ID_0003", BIOLOGICAL_SEX] == FEMALE
        assert result.loc["ID_0003", AGE_AT_INITIAL_DIAGNOSIS] == "19"

    def test_fallback_predicate_is_used_for_both_variables(self, sparql_endpoint):
        """Test that either variable of a multi-column query uses the fallback.

        A substitution that is only applied to the first variable would leave the
        second placeholder in the query, which no store would accept.
        """
        collect([BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS], query_type="multi_column")

        posted_query = sparql_endpoint[0]
        assert "PLACEHOLDER" not in posted_query
        assert posted_query.count("dbo:has_column") == 2


class TestFallbackAgreesWithTheSchema:
    """Test that both modes observe the same data in an annotated graph."""

    def test_single_column_results_are_identical(self, sparql_endpoint):
        """Test that either mode collects the same single-column data.

        The synthetic graph carries both the schema's structure and the annotation,
        so the two modes reach the very same value nodes. Any difference between their
        results would therefore originate from the queries rather than from the data,
        which is precisely what must not happen when a node switches mode.
        """
        variables = [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS]

        fallback_result = collect(variables, use_schema=False)
        schema_result = collect(variables, use_schema=True)

        # The two modes did ask for the data differently, so their agreement is one
        # of results rather than one of queries
        fallback_query, schema_query = sparql_endpoint[0], sparql_endpoint[2]
        assert fallback_query != schema_query
        assert "dbo:has_column" in fallback_query
        assert "sio:SIO_000255/sio:SIO_000008" in schema_query

        pd.testing.assert_frame_equal(fallback_result, schema_result)

    def test_multi_column_results_are_identical(self, sparql_endpoint):
        """Test that either mode collects the same multi-column data.

        The multi-column query resolves the identity of two records in a single query,
        which means that a mode-dependent predicate path could pair the values of one
        record with those of another without the results looking obviously wrong.
        """
        variables = [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS]

        fallback_result = collect(variables, query_type="multi_column")
        schema_result = collect(variables, query_type="multi_column", use_schema=True)

        # Either mode fetched both variables through a query of its own shape
        fallback_query, schema_query = sparql_endpoint
        assert fallback_query != schema_query
        assert fallback_query.count("dbo:has_column") == 2
        assert schema_query.count("sio:SIO_000255/sio:SIO_000008") == 2

        pd.testing.assert_frame_equal(fallback_result, schema_result)


class TestCustomVariableProperty:
    """Test that a custom variable property is honoured in the fallback mode."""

    def test_custom_property_replaces_the_default(self, sparql_endpoint):
        """Test that a given property is the one that the query is built with.

        A node whose annotation uses another predicate configures it through the
        variable property; silently querying dbo:has_column instead would make that
        configuration meaningless.
        """
        collect([BIOLOGICAL_SEX], variable_property=UNUSED_VARIABLE_PROPERTY)

        posted_query = sparql_endpoint[0]
        assert f"?patient {UNUSED_VARIABLE_PROPERTY} ?sub_class_type" in posted_query
        assert "dbo:has_column" not in posted_query

    def test_custom_property_that_the_graph_does_not_use_yields_no_data(
        self, sparql_endpoint
    ):
        """Test that a property which the graph does not describe yields no data.

        An absence of data is an observation rather than an error; the variable is
        simply not part of the collected data, just as it would not be on a node whose
        graph has not been annotated.
        """
        result = collect([BIOLOGICAL_SEX], variable_property=UNUSED_VARIABLE_PROPERTY)

        assert result.empty

    def test_custom_property_is_honoured_by_the_multi_column_query(
        self, sparql_endpoint
    ):
        """Test that the multi-column query honours a custom property as well.

        The property is substituted once per variable, so a multi-column query would
        otherwise be able to fall back to a predicate that was never configured.
        """
        result = collect(
            [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS],
            query_type="multi_column",
            variable_property=UNUSED_VARIABLE_PROPERTY,
        )

        posted_query = sparql_endpoint[0]
        assert posted_query.count(UNUSED_VARIABLE_PROPERTY) == 2
        assert "dbo:has_column" not in posted_query
        assert result.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
