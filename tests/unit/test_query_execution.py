"""
Unit tests that execute the generated SPARQL queries against a synthetic RDF graph.

The graph mirrors the structure that the Triplifier produces and that the AYA cancer
schema describes, which allows the queries to be verified without an RDF-store:

    record ─sio:SIO_000673─> identifier node ─dbo:has_cell/dbo:has_value─> "ID_0001"
    record ─sio:SIO_000255─> characteristics ─sio:SIO_000008─> variable node
                                                     └─dbo:has_cell─> cell ─> value
    variable node ─sio:SIO_000253─> PROM ─sio:SIO_000233─> variable node

The following records are described:

- ID_0001: a record of a flat table (biological sex and age) and a record of a second
  table that holds the same identifier (the number of days since diagnosis);
- ID_0002: a record of a flat table (biological sex and age) and a record of a linked
  table whose foreign key refers to that record (the number of days since diagnosis);
- a record without an identifier (biological sex).

Type inference is not emulated; the classes that a store would infer through
owl:equivalentClass and rdfs:subClassOf are asserted directly.
"""

import sys

from importlib import import_module
from pathlib import Path

import pytest

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# The package re-exports the collect_sparql_data function under the name of its module,
# which is why the module itself is imported explicitly
data_collection = import_module("vantage6_strongaya_rdf.collect_sparql_data")

rdflib = pytest.importorskip(
    "rdflib", reason="rdflib is required to execute the generated SPARQL queries"
)

# Classes and predicates of the AYA cancer schema that the synthetic graph describes
BIOLOGICAL_SEX = "ncit:C28421"
AGE_AT_INITIAL_DIAGNOSIS = "ncit:C156420"
TIME_PROM_RECORDING = "ncit:C192402"
MALE = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C20197"
FEMALE = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C16576"

# URI of the record that holds no identifier
RECORD_WITHOUT_IDENTIFIER = "http://data.local/rdf/data/record_without_identifier"

SYNTHETIC_GRAPH = """
@prefix dbo: <http://um-cds/ontologies/databaseontology/> .
@prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sio: <http://semanticscience.org/resource/> .
@prefix data: <http://data.local/rdf/data/> .

# ---------- Subclasses that a store would infer ----------
ncit:C20197 rdfs:subClassOf ncit:C28421 .
ncit:C16576 rdfs:subClassOf ncit:C28421 .

# ---------- ID_0001; a record of the primary table ----------
data:record_one sio:SIO_000673 data:record_one_id .
data:record_one_id a ncit:C25364 ;
                   dbo:has_cell data:record_one_id_cell .
data:record_one_id_cell dbo:has_value "ID_0001" .

data:record_one sio:SIO_000255 data:record_one_characteristics .
data:record_one_characteristics sio:SIO_000008 data:record_one_sex ,
                                               data:record_one_age .
data:record_one_sex a ncit:C28421 ;
                    dbo:has_cell data:record_one_sex_cell .
data:record_one_sex_cell a ncit:C20197 ;
                         dbo:has_value "male" .
data:record_one_age a ncit:C156420 ;
                    dbo:has_cell data:record_one_age_cell .
data:record_one_age_cell dbo:has_value "27" .

# ---------- ID_0001; a record of a second table that repeats the identifier ----------
data:record_two sio:SIO_000673 data:record_two_id .
data:record_two_id a ncit:C25364 ;
                   dbo:has_cell data:record_two_id_cell .
data:record_two_id_cell dbo:has_value "ID_0001" .

data:record_two sio:SIO_000255 data:record_two_characteristics .
data:record_two_characteristics sio:SIO_000008 data:record_two_gender .
data:record_two_gender a ncit:C158277 ;
                       dbo:has_cell data:record_two_gender_cell ;
                       sio:SIO_000253 data:record_two_prom .
data:record_two_gender_cell dbo:has_value "man" .
data:record_two_prom a ncit:C177377 ;
                     sio:SIO_000233 data:record_two_days .
data:record_two_days a ncit:C192402 ;
                     dbo:has_cell data:record_two_days_cell .
data:record_two_days_cell dbo:has_value "42" .

# ---------- ID_0002; a record of the primary table ----------
data:record_three sio:SIO_000673 data:record_three_id .
data:record_three_id a ncit:C25364 ;
                     dbo:has_cell data:record_three_id_cell .
data:record_three_id_cell dbo:has_value "ID_0002" .

data:record_three sio:SIO_000255 data:record_three_characteristics .
data:record_three_characteristics sio:SIO_000008 data:record_three_sex ,
                                                 data:record_three_age .
data:record_three_sex a ncit:C28421 ;
                      dbo:has_cell data:record_three_sex_cell .
data:record_three_sex_cell a ncit:C16576 ;
                           dbo:has_value "female" .
data:record_three_age a ncit:C156420 ;
                      dbo:has_cell data:record_three_age_cell .
data:record_three_age_cell dbo:has_value "31" .

# ---------- ID_0002; a record of a linked table that refers to it ----------
data:record_four sio:SIO_000673 data:record_four_id .
data:record_four_id a ncit:C25364 ;
                    dbo:has_cell data:record_four_id_cell ;
                    dbo:fk_refers_to data:record_three_id .
data:record_four_id_cell dbo:has_value "ROW_0009" .

data:record_four sio:SIO_000255 data:record_four_characteristics .
data:record_four_characteristics sio:SIO_000008 data:record_four_gender .
data:record_four_gender a ncit:C158277 ;
                        dbo:has_cell data:record_four_gender_cell ;
                        sio:SIO_000253 data:record_four_prom .
data:record_four_gender_cell dbo:has_value "woman" .
data:record_four_prom a ncit:C177377 ;
                      sio:SIO_000233 data:record_four_days .
data:record_four_days a ncit:C192402 ;
                      dbo:has_cell data:record_four_days_cell .
data:record_four_days_cell dbo:has_value "7" .

# ---------- A record whose identifier cannot be retrieved ----------
data:record_without_identifier sio:SIO_000255 data:record_five_characteristics .
data:record_five_characteristics sio:SIO_000008 data:record_five_sex .
data:record_five_sex a ncit:C28421 ;
                     dbo:has_cell data:record_five_sex_cell .
data:record_five_sex_cell a ncit:C16576 ;
                          dbo:has_value "female" .
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
    an empty string for the variables that a solution leaves unbound.
    """
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


def collect(variables, query_type="single_column"):
    """
    Collect data through the library, as an algorithm would.

    :param variables: Variables (or class codes) to collect
    :param query_type: The type of query to use
    :return: pd.DataFrame with the collected data
    """
    return data_collection.collect_sparql_data(
        variables,
        query_type=query_type,
        endpoint="http://localhost:7200/repositories/synthetic",
        use_schema=True,
    )


class TestSingleColumnQuery:
    """Test the single-column query against the synthetic graph."""

    def test_categorical_variable(self, sparql_endpoint):
        """Test that a categorical variable is collected as its subclass."""
        result = collect([BIOLOGICAL_SEX])

        assert set(result["patient_id"]) == {
            "ID_0001",
            "ID_0002",
            RECORD_WITHOUT_IDENTIFIER,
        }
        values = dict(zip(result["patient_id"], result[BIOLOGICAL_SEX]))
        assert values["ID_0001"] == MALE
        assert values["ID_0002"] == FEMALE

    def test_continuous_variable(self, sparql_endpoint):
        """Test that a continuous variable is collected as its value."""
        result = collect([AGE_AT_INITIAL_DIAGNOSIS])

        values = dict(zip(result["patient_id"], result[AGE_AT_INITIAL_DIAGNOSIS]))
        assert values == {"ID_0001": "27", "ID_0002": "31"}

    def test_variable_within_an_intermediate_class(self, sparql_endpoint):
        """Test that a variable that is nested in the PROM container is collected.

        The path towards such a variable used to omit the hop that reaches the
        container, which meant that no data was found at all.
        """
        result = collect([TIME_PROM_RECORDING])

        values = dict(zip(result["patient_id"], result[TIME_PROM_RECORDING]))
        assert values == {"ID_0001": "42", "ID_0002": "7"}

    def test_variables_of_different_tables_are_merged(self, sparql_endpoint):
        """Test that variables of separate tables are merged on the patient."""
        result = collect([BIOLOGICAL_SEX, TIME_PROM_RECORDING])

        result = result.set_index("patient_id")
        assert result.loc["ID_0001", TIME_PROM_RECORDING] == "42"
        assert result.loc["ID_0001", BIOLOGICAL_SEX] == MALE
        assert result.loc["ID_0002", TIME_PROM_RECORDING] == "7"
        assert result.loc["ID_0002", BIOLOGICAL_SEX] == FEMALE

    def test_record_without_an_identifier_is_retained(self, sparql_endpoint):
        """Test that a record whose identifier cannot be retrieved is not dropped.

        Such a record used to disappear from the results altogether, which meant that
        it was not even counted as missing data.
        """
        result = collect([BIOLOGICAL_SEX])

        assert RECORD_WITHOUT_IDENTIFIER in set(result["patient_id"])
        assert len(result) == 3

        values = dict(zip(result["patient_id"], result[BIOLOGICAL_SEX]))
        assert values[RECORD_WITHOUT_IDENTIFIER] == FEMALE

    def test_record_without_an_identifier_is_reported(self, sparql_endpoint, capsys):
        """Test that a record without an identifier is reported as a warning."""
        collect([BIOLOGICAL_SEX])

        logged = capsys.readouterr().out
        assert "warn > Could not retrieve the identifier of 1 record(s)" in logged

    def test_record_of_a_linked_table_is_identified_by_its_reference(
        self, sparql_endpoint
    ):
        """Test that a foreign key resolves to the identifier that it refers to.

        The record of the linked table holds its own row identifier ("ROW_0009"),
        which would keep it from lining up with the records of the primary table.
        """
        result = collect([TIME_PROM_RECORDING])

        assert "ROW_0009" not in set(result["patient_id"])
        assert set(result["patient_id"]) == {"ID_0001", "ID_0002"}

    def test_unknown_variable_yields_no_data(self, sparql_endpoint):
        """Test that a variable that the graph does not describe yields no data.

        An absence of data is an observation rather than an error; the variable is
        simply not part of the collected data.
        """
        result = collect(["ncit:C0123456789"])

        assert result.empty


class TestMultiColumnQuery:
    """Test the multi-column query against the synthetic graph."""

    def test_variables_of_the_same_table(self, sparql_endpoint):
        """Test that two variables of a single table are fetched together."""
        result = collect(
            [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS], query_type="multi_column"
        )

        result = result.set_index("patient_id")
        assert set(result.index) == {"ID_0001", "ID_0002"}
        assert result.loc["ID_0001", BIOLOGICAL_SEX] == MALE
        assert result.loc["ID_0001", AGE_AT_INITIAL_DIAGNOSIS] == "27"
        assert result.loc["ID_0002", BIOLOGICAL_SEX] == FEMALE
        assert result.loc["ID_0002", AGE_AT_INITIAL_DIAGNOSIS] == "31"

    def test_variables_of_separate_tables(self, sparql_endpoint):
        """Test that two variables of separate tables are joined on the patient.

        The identity of the two records used to be resolved in a group that held
        nothing but filters, in which the identifiers are out of scope; the query
        therefore never returned anything at all.
        """
        result = collect(
            [BIOLOGICAL_SEX, TIME_PROM_RECORDING], query_type="multi_column"
        )

        result = result.set_index("patient_id")
        assert set(result.index) == {"ID_0001", "ID_0002"}

        # ID_0001's records hold the same identifier; ID_0002's records are linked
        # through a foreign key
        assert result.loc["ID_0001", BIOLOGICAL_SEX] == MALE
        assert result.loc["ID_0001", TIME_PROM_RECORDING] == "42"
        assert result.loc["ID_0002", BIOLOGICAL_SEX] == FEMALE
        assert result.loc["ID_0002", TIME_PROM_RECORDING] == "7"

    def test_values_of_separate_patients_are_not_combined(self, sparql_endpoint):
        """Test that the values of a record are not paired with those of another."""
        result = collect(
            [BIOLOGICAL_SEX, TIME_PROM_RECORDING], query_type="multi_column"
        )

        # The record without an identifier holds no time of PROM recording, so it
        # cannot - and must not - be paired with the recording of another record
        assert RECORD_WITHOUT_IDENTIFIER not in set(result["patient_id"])
        assert len(result) == 2

    def test_multi_column_requires_two_variables(self, sparql_endpoint):
        """Test that the multi-column query rejects any other number of variables."""
        from vantage6.algorithm.tools.exceptions import AlgorithmError

        with pytest.raises(AlgorithmError):
            collect([BIOLOGICAL_SEX], query_type="multi_column")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
