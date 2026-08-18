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


# ---------- Records whose identifiers are numeric ----------
# A dataset's identifiers are often numeric; they used to be converted to numbers, which
# made the type of the identifier depend on the values of a single variable.
NUMERIC_IDENTIFIER_GRAPH = """
@prefix dbo: <http://um-cds/ontologies/databaseontology/> .
@prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sio: <http://semanticscience.org/resource/> .
@prefix data: <http://data.local/rdf/data/> .

ncit:C20197 rdfs:subClassOf ncit:C28421 .
ncit:C16576 rdfs:subClassOf ncit:C28421 .

# ---------- Identifier "1"; holds both variables ----------
data:record_one sio:SIO_000673 data:record_one_id .
data:record_one_id a ncit:C25364 ;
                   dbo:has_cell data:record_one_id_cell .
data:record_one_id_cell dbo:has_value "1" .

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

# ---------- Identifier "2"; holds the biological sex only ----------
data:record_two sio:SIO_000673 data:record_two_id .
data:record_two_id a ncit:C25364 ;
                   dbo:has_cell data:record_two_id_cell .
data:record_two_id_cell dbo:has_value "2" .

data:record_two sio:SIO_000255 data:record_two_characteristics .
data:record_two_characteristics sio:SIO_000008 data:record_two_sex .
data:record_two_sex a ncit:C28421 ;
                    dbo:has_cell data:record_two_sex_cell .
data:record_two_sex_cell a ncit:C16576 ;
                         dbo:has_value "female" .

# ---------- Identifier "10"; holds both variables ----------
data:record_three sio:SIO_000673 data:record_three_id .
data:record_three_id a ncit:C25364 ;
                     dbo:has_cell data:record_three_id_cell .
data:record_three_id_cell dbo:has_value "10" .

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

# ---------- A record whose identifier cannot be retrieved ----------
data:record_without_identifier sio:SIO_000255 data:record_four_characteristics .
data:record_four_characteristics sio:SIO_000008 data:record_four_sex ,
                                                data:record_four_age .
data:record_four_sex a ncit:C28421 ;
                     dbo:has_cell data:record_four_sex_cell .
data:record_four_sex_cell a ncit:C20197 ;
                          dbo:has_value "male" .
data:record_four_age a ncit:C156420 ;
                     dbo:has_cell data:record_four_age_cell .
data:record_four_age_cell dbo:has_value "45" .
"""

# ---------- Records that do not all hold the same variables ----------
PARTIAL_COVERAGE_GRAPH = """
@prefix dbo: <http://um-cds/ontologies/databaseontology/> .
@prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sio: <http://semanticscience.org/resource/> .
@prefix data: <http://data.local/rdf/data/> .

ncit:C20197 rdfs:subClassOf ncit:C28421 .

# ---------- ID_0001; holds both the biological sex and the age ----------
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

# ---------- ID_0002; holds the biological sex only ----------
data:record_two sio:SIO_000673 data:record_two_id .
data:record_two_id a ncit:C25364 ;
                   dbo:has_cell data:record_two_id_cell .
data:record_two_id_cell dbo:has_value "ID_0002" .

data:record_two sio:SIO_000255 data:record_two_characteristics .
data:record_two_characteristics sio:SIO_000008 data:record_two_sex .
data:record_two_sex a ncit:C28421 ;
                    dbo:has_cell data:record_two_sex_cell .
data:record_two_sex_cell a ncit:C20197 ;
                         dbo:has_value "male" .
"""

# ---------- Records whose cells hold no meaningful value ----------
# The Triplifier writes an absent value as the string "NULL"; a dataset may in addition
# use its own notation for missing data, such as "-99".
NULL_VALUE_GRAPH = """
@prefix dbo: <http://um-cds/ontologies/databaseontology/> .
@prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> .
@prefix sio: <http://semanticscience.org/resource/> .
@prefix data: <http://data.local/rdf/data/> .

# ---------- ID_0001; holds an age ----------
data:record_one sio:SIO_000673 data:record_one_id .
data:record_one_id a ncit:C25364 ;
                   dbo:has_cell data:record_one_id_cell .
data:record_one_id_cell dbo:has_value "ID_0001" .

data:record_one sio:SIO_000255 data:record_one_characteristics .
data:record_one_characteristics sio:SIO_000008 data:record_one_age .
data:record_one_age a ncit:C156420 ;
                    dbo:has_cell data:record_one_age_cell .
data:record_one_age_cell dbo:has_value "27" .

# ---------- ID_0002; holds a NULL age ----------
data:record_two sio:SIO_000673 data:record_two_id .
data:record_two_id a ncit:C25364 ;
                   dbo:has_cell data:record_two_id_cell .
data:record_two_id_cell dbo:has_value "ID_0002" .

data:record_two sio:SIO_000255 data:record_two_characteristics .
data:record_two_characteristics sio:SIO_000008 data:record_two_age .
data:record_two_age a ncit:C156420 ;
                    dbo:has_cell data:record_two_age_cell .
data:record_two_age_cell dbo:has_value "NULL" .

# ---------- ID_0003; holds an age of the dataset's own missing notation ----------
data:record_three sio:SIO_000673 data:record_three_id .
data:record_three_id a ncit:C25364 ;
                     dbo:has_cell data:record_three_id_cell .
data:record_three_id_cell dbo:has_value "ID_0003" .

data:record_three sio:SIO_000255 data:record_three_characteristics .
data:record_three_characteristics sio:SIO_000008 data:record_three_age .
data:record_three_age a ncit:C156420 ;
                      dbo:has_cell data:record_three_age_cell .
data:record_three_age_cell dbo:has_value "-99" .
"""

# ---------- A record that holds more than one value for a single variable ----------
REPEATED_MEASURE_GRAPH = """
@prefix dbo: <http://um-cds/ontologies/databaseontology/> .
@prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> .
@prefix sio: <http://semanticscience.org/resource/> .
@prefix data: <http://data.local/rdf/data/> .

# ---------- ID_0001; the PROM container holds two recordings ----------
data:record_one sio:SIO_000673 data:record_one_id .
data:record_one_id a ncit:C25364 ;
                   dbo:has_cell data:record_one_id_cell .
data:record_one_id_cell dbo:has_value "ID_0001" .

data:record_one sio:SIO_000255 data:record_one_characteristics .
data:record_one_characteristics sio:SIO_000008 data:record_one_gender .
data:record_one_gender a ncit:C158277 ;
                       dbo:has_cell data:record_one_gender_cell ;
                       sio:SIO_000253 data:record_one_prom .
data:record_one_gender_cell dbo:has_value "man" .
data:record_one_prom a ncit:C177377 ;
                     sio:SIO_000233 data:record_one_first_days ,
                                    data:record_one_second_days .
data:record_one_first_days a ncit:C192402 ;
                           dbo:has_cell data:record_one_first_days_cell .
data:record_one_first_days_cell dbo:has_value "42" .
data:record_one_second_days a ncit:C192402 ;
                            dbo:has_cell data:record_one_second_days_cell .
data:record_one_second_days_cell dbo:has_value "84" .
"""


def _execute_against(graph_definition: str, monkeypatch) -> list:
    """
    Let the library post its queries to a graph rather than to an RDF-store.

    The results are returned the way that an RDF-store returns them; as strings, with
    an empty string for the variables that a solution leaves unbound.

    :param graph_definition: The graph to execute the queries against, as Turtle
    :param monkeypatch: The pytest monkeypatch fixture
    :return: list that collects the queries that the library posts
    """
    synthetic_graph = rdflib.Graph()
    synthetic_graph.parse(data=graph_definition, format="turtle")

    posted_queries = []

    def execute_query(endpoint: str, query: str, **kwargs):
        posted_queries.append(query)
        result = synthetic_graph.query(query)
        return [
            {
                str(variable): ("" if value is None else str(value))
                for variable, value in zip(result.vars, row)
            }
            for row in result
        ]

    monkeypatch.setattr(data_collection, "post_sparql_query", execute_query)
    return posted_queries


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


@pytest.fixture
def numeric_identifier_endpoint(monkeypatch):
    """Execute the library's queries against the graph with numeric identifiers."""
    return _execute_against(NUMERIC_IDENTIFIER_GRAPH, monkeypatch)


@pytest.fixture
def partial_coverage_endpoint(monkeypatch):
    """Execute the library's queries against the graph of unequal coverage."""
    return _execute_against(PARTIAL_COVERAGE_GRAPH, monkeypatch)


@pytest.fixture
def null_value_endpoint(monkeypatch):
    """Execute the library's queries against the graph that holds NULL cells."""
    return _execute_against(NULL_VALUE_GRAPH, monkeypatch)


@pytest.fixture
def repeated_measure_endpoint(monkeypatch):
    """Execute the library's queries against the graph of repeated measures."""
    return _execute_against(REPEATED_MEASURE_GRAPH, monkeypatch)


def collect(variables, query_type="single_column", **kwargs):
    """
    Collect data through the library, as an algorithm would.

    :param variables: Variables (or class codes) to collect
    :param query_type: The type of query to use
    :param kwargs: Any other argument of collect_sparql_data
    :return: pd.DataFrame with the collected data
    """
    return data_collection.collect_sparql_data(
        variables,
        query_type=query_type,
        endpoint="http://localhost:7200/repositories/synthetic",
        use_schema=True,
        **kwargs,
    )


def missing_values_of(result, variable: str) -> int:
    """
    Retrieve the number of missing values that was determined for a variable.

    :param result: pd.DataFrame that the library returned
    :param variable: The variable to retrieve the number of missing values of
    :return: int holding the number of missing values
    """
    return result.attrs["stats"]["missing_values"]["value"][variable]


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


class TestNumericIdentifiers:
    """Test the collection of records whose identifiers are numeric."""

    def test_variables_are_merged_when_identifiers_are_numeric(
        self, numeric_identifier_endpoint
    ):
        """Test that variables are merged when a dataset's identifiers are numeric.

        The identifiers used to be converted to numbers wherever they happened to be
        numeric, which made the type of the identifier depend on the values of a single
        variable: a variable whose records all hold an identifier yielded numbers,
        whereas a variable that holds a record without an identifier yielded text.
        Merging the two then failed altogether, which meant that no data was collected
        at all rather than that a single record was affected.
        """
        result = collect([BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS])

        assert set(result["patient_id"]) == {
            "1",
            "2",
            "10",
            RECORD_WITHOUT_IDENTIFIER,
        }

        result = result.set_index("patient_id")
        assert result.loc["1", BIOLOGICAL_SEX] == MALE
        assert result.loc["1", AGE_AT_INITIAL_DIAGNOSIS] == "27"
        assert result.loc["10", AGE_AT_INITIAL_DIAGNOSIS] == "31"
        assert result.loc[RECORD_WITHOUT_IDENTIFIER, AGE_AT_INITIAL_DIAGNOSIS] == "45"

        # The record that holds no age is retained; its absence is an observation
        assert pd.isna(result.loc["2", AGE_AT_INITIAL_DIAGNOSIS])

    def test_identifiers_are_collected_as_text(self, numeric_identifier_endpoint):
        """Test that identifiers are text, whichever values a dataset holds.

        An identifier is a label rather than a quantity, and it may hold characters in
        one dataset and digits only in another; representing it as text consistently is
        what allows the variables of separate tables to be merged.
        """
        result = collect([AGE_AT_INITIAL_DIAGNOSIS])

        assert result["patient_id"].map(type).eq(str).all()

    def test_records_are_ordered_naturally(self, numeric_identifier_endpoint):
        """Test that numeric identifiers are ordered by their value, not their text.

        A plain sort of text would place "10" before "2", which would make the order of
        the collected data counter-intuitive for a numerically identified dataset.
        """
        result = collect([BIOLOGICAL_SEX])

        assert list(result["patient_id"]) == [
            "1",
            "2",
            "10",
            RECORD_WITHOUT_IDENTIFIER,
        ]

    def test_ordering_combines_identifiers_of_any_shape(
        self, numeric_identifier_endpoint
    ):
        """Test that a record without an identifier does not break the ordering.

        Such a record is identified by its own URI, which means that a single variable
        can hold both numeric identifiers and a URI; comparing the two used to raise a
        TypeError while sorting.
        """
        result = collect([AGE_AT_INITIAL_DIAGNOSIS])

        assert list(result["patient_id"]) == ["1", "10", RECORD_WITHOUT_IDENTIFIER]


class TestUnequalCoverage:
    """Test the collection of patients that do not all hold the same variables."""

    def test_single_column_retains_a_patient_without_a_value(
        self, partial_coverage_endpoint
    ):
        """Test that a patient is retained for a variable that they hold no value for.

        An absence of data is an observation rather than an error, which is why the
        patient remains part of the collected data and counts towards its size.
        """
        result = collect([BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS])

        assert set(result["patient_id"]) == {"ID_0001", "ID_0002"}

        result = result.set_index("patient_id")
        assert result.loc["ID_0002", BIOLOGICAL_SEX] == MALE
        assert pd.isna(result.loc["ID_0002", AGE_AT_INITIAL_DIAGNOSIS])

    def test_multi_column_keeps_the_patients_that_hold_both_variables(
        self, partial_coverage_endpoint
    ):
        """Test that the multi-column query only yields patients that hold both values.

        Both variables are fetched within a single query, which joins them on the
        patient; a patient that holds one of the two is therefore not part of the
        result, unlike the single-column query that merges the variables afterwards.
        This difference is documented in the README.
        """
        result = collect(
            [BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS], query_type="multi_column"
        )

        assert set(result["patient_id"]) == {"ID_0001"}

    def test_absent_value_is_counted_as_a_missing_value(
        self, partial_coverage_endpoint
    ):
        """Test that a value that no record holds is counted as a missing value.

        An absence of data is only a usable observation when it is counted; the
        missing-value statistics of an algorithm are determined from this count, so a
        patient that holds no value has to be reflected in it.
        """
        result = collect([BIOLOGICAL_SEX, AGE_AT_INITIAL_DIAGNOSIS])

        assert missing_values_of(result, AGE_AT_INITIAL_DIAGNOSIS) == 1
        assert missing_values_of(result, BIOLOGICAL_SEX) == 0


class TestMissingValueNotations:
    """Test the collection of cells that hold no meaningful value."""

    def test_null_cells_are_collected_as_missing_values(self, null_value_endpoint):
        """Test that a cell that the Triplifier wrote as "NULL" is missing data.

        The Triplifier represents an absent value as the string "NULL", which would
        otherwise be treated as an ordinary value of a categorical variable.
        """
        result = collect([AGE_AT_INITIAL_DIAGNOSIS])

        assert set(result["patient_id"]) == {"ID_0001", "ID_0002", "ID_0003"}

        values = dict(zip(result["patient_id"], result[AGE_AT_INITIAL_DIAGNOSIS]))
        assert values["ID_0001"] == "27"
        assert pd.isna(values["ID_0002"])
        assert missing_values_of(result, AGE_AT_INITIAL_DIAGNOSIS) == 1

    def test_missing_data_notation_is_collected_as_missing_values(
        self, null_value_endpoint
    ):
        """Test that a dataset's own notation for missing data is honoured.

        A dataset may denote missing data with a value of its own, such as "-99"; such
        a value has to be counted as missing data alongside the Triplifier's "NULL",
        rather than being analysed as if it were a measurement.
        """
        result = collect([AGE_AT_INITIAL_DIAGNOSIS], missing_data_notation="-99")

        values = dict(zip(result["patient_id"], result[AGE_AT_INITIAL_DIAGNOSIS]))
        assert values["ID_0001"] == "27"
        assert pd.isna(values["ID_0002"])
        assert pd.isna(values["ID_0003"])
        assert missing_values_of(result, AGE_AT_INITIAL_DIAGNOSIS) == 2


class TestRepeatedMeasures:
    """Test the collection of a record that holds several values for one variable."""

    def test_repeated_measures_are_collected_as_a_single_value(
        self, repeated_measure_endpoint
    ):
        """Test that a repeated measure yields one value per patient.

        Both query templates sample a single value per patient, which means that a
        record holding several recordings of the same variable - two PROM recordings,
        for instance - is represented by one of them. This is the library's current
        behaviour rather than a deliberate design decision; extracting every recording
        separately requires the queries to return one row per recording, which is a
        change of the result's shape that this test will report on.
        """
        result = collect([TIME_PROM_RECORDING])

        assert list(result["patient_id"]) == ["ID_0001"]
        assert result[TIME_PROM_RECORDING].iloc[0] in {"42", "84"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
