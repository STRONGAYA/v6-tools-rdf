# STRONG AYA's RDF Vantage6 tools

<p align="center">
<a href="https://github.com/STRONGAYA/v6-tools-rdf/workflows/"><img alt="Test status" src="https://github.com/STRONGAYA/v6-tools-rdf/workflows/Test%20Suite/badge.svg"></a>
<a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img alt="Licence: Apache 2.0" src="https://img.shields.io/badge/Licence-Apache%202.0-blue.svg"></a>
<br>
<a href="https://github.com/vantage6/vantage6/"><img alt="Vantage6 4.11, 4.12" src="https://img.shields.io/badge/vantage6- 4.11 | 4.12-blue.svg"></a>
<a href="https://github.com/MaastrichtU-CDS/Flyover"><img alt="Flyover version 2.0+" src="https://img.shields.io/badge/Flyover%20Version-2.0+-purple"></a>
<a href="https://strongaya.eu/wp-content/uploads/2025/07/algorithm_review_guidelines.pdf"><img alt="STRONG AYA Algorithm Guideline Conformity: v1.1.0 Approved" src="https://img.shields.io/badge/STRONG%20AYA%20Algorithm%20Guideline%20Conformity-v1.1.0%20approved-brightgreen">
<br>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://flake8.pycqa.org/"><img alt="Linting: flake8" src="https://img.shields.io/badge/linting-flake8-informational"></a>
<a href="http://mypy-lang.org/"><img alt="Type checking: mypy" src="https://img.shields.io/badge/type%20checking-mypy-informational"></a>
<a href="https://github.com/PyCQA/bandit"><img alt="Security: bandit" src="https://img.shields.io/badge/security-bandit-informational"></a>
<a href="https://github.com/pyupio/safety"><img alt="Security: safety" src="https://img.shields.io/badge/security-safety-informational"></a>
</p>

<!--
To show the approved badge instead, use:
<a href="https://strongaya.eu/wp-content/uploads/2025/07/algorithm_review_guidelines.pdf"><img alt="STRONG AYA Algorithm Guideline Conformity: vx.x.x Approved" src="https://img.shields.io/badge/STRONG%20AYA%20Algorithm%20Guideline%20Conformity-vx.x.x%20approved-brightgreen">

To show the pending badge instead, use:
<a href="https://strongaya.eu/wp-content/uploads/2025/07/algorithm_review_guidelines.pdf"><img alt="STRONG AYA Algorithm Guideline Conformity: vx.x.x Pending" src="https://img.shields.io/badge/STRONG%20AYA%20Algorithm%20Guideline%20Conformity-vx.x.x%20pending-yellow"></a>
-->

# Purpose of this repository

This repository contains resource description framework (RDF) functionalities and tools for the STRONG AYA project.
They are designed to be used with the Vantage6 framework for federated analytics and learning
and are intended to facilitate and simplify the development of Vantage6 algorithms.
The SPARQL queries and RDF functionalities are designed to be used in conjunction with the Flyover and Triplifier tools.

The code in this repository is available as a Python library here on GitHub or through direct reference with `pip`.

# Structure of the repository

The various functions are organised in different sections, consisting of:

- **RDF Data Collection**: Functions to formulate and execute a SPARQL query on an RDF/SPARQL endpoint;
- **Data Processing**: Functions to process the output of an RDF/SPARQL endpoint (e.g. determine missing values, extract
  associated subclasses);
- **Schema Loader**: Functions to load the AYA cancer JSON-LD schema from bundled resources or remote URL;
- **Schema Parser**: Functions to parse the schema and build dynamic SPARQL predicate paths;
- **Query Templates**: SPARQL query templates that the SPARQL data collection section uses (supports both single-column and multi-column queries)

# Usage

The library provides functions that can be included in a Vantage6 algorithm as the algorithm developer sees fit.
The functions are designed to be modular and can be used independently or in combination with other functions.

The library can be included in your Vantage6 algorithm by listing it in the `requirements.txt` and `setup.py` file of
your
algorithm.

## Including the library in your Vantage6 algorithm

For the `requirements.txt` file, you can add the following line to the file:

```
git+https://github.com/STRONGAYA/v6-tools-rdf.git@v1.1.0
```

For the `setup.py` file, you can add the following line to the `install_requires` list:

```python
        "vantage6-strongaya-rdf @ git+https://github.com/STRONGAYA/v6-tools-rdf.git@v1.1.0",
```

The algorithm's `setup.py`, particularly the `install_requirements`, section file should then look something like this:

```python
from os import path
from codecs import open
from setuptools import setup, find_packages

# We are using a README.md, if you do not have this in your folder, simply replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
    name='v6-not-an-actual-algorithm',
    version="1.0.1",
    description='Fictive Vantage6 algorithm that performs general statistics computation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/STRONGAYA/v6-not-an-actual-algorithm',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'vantage6-algorithm-tools',
        'numpy',
        'pandas',
        "vantage6-strongaya-rdf @ git+https://github.com/STRONGAYA/v6-tools-rdf.git@v1.1.0"
        # other dependencies
    ]
)
```

## Central (aggregating) example

The functions included in this library focus on extracting RDF data from a SPARQL endpoint.
It is not recommended to use these functions in the central (aggregating) section of a Vantage6 algorithm.

## Node or local (participating) example

Example usage of the SPARQL data collection function in a node (participating) section of a Vantage6 algorithm:

```python
# General federated algorithm functions
from vantage6_strongaya_general.miscellaneous import safe_log
from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data


def partial_general_statistics(variables_to_extract: dict) -> dict:
    """
    Execute the partial algorithm for some modelling using RDF data.

    Args:
        variables_to_extract (list): List of variables to extract.

    Returns:
        dict: A dictionary containing the computed general statistics.
    """
    safe_log("info", "Executing partial algorithm for some modelling using RDF data.")

    # Set datatypes for each variable
    df = collect_sparql_data(list(variables_to_extract.keys()), query_type="single_column",
                             endpoint="http://localhost:7200/repositories/userRepo",
                             )

    # Ensure that the desired privacy measures are applied

    # Do some modelling of the data

    return result
```

## Schema-Based Dynamic SPARQL Query Generation

The library now supports dynamic SPARQL predicate path building based on the AYA cancer JSON-LD schema. This feature allows for more flexible and schema-aware querying of RDF data.

### Using Schema-Based Queries

To use schema-based query generation, set `use_schema=True`. When using schema-based queries, the `variable_property` parameter is optional as the predicate paths are built automatically from the schema:

```python
from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data

# Fetch data with schema-based predicate paths
# Note: variable_property is optional when use_schema=True
df = collect_sparql_data(
    variables_to_extract=['age_at_initial_diagnosis', 'gender'],
    query_type="single_column",
    endpoint="http://localhost:7200/repositories/userRepo",
    use_schema=True  # Enable schema-based path generation
)
```

Variables can be requested by their name in the schema (`'age_at_initial_diagnosis'`) or by
their class code (`'ncit:C156420'`).

The predicate paths follow the structure that the schema describes as an ordered sequence,
for instance `sio:SIO_000255/sio:SIO_000008`. Variables that are nested within an
intermediate class - the PROM, EHR and HCPROM containers - are resolved to the route
towards the container and the hop into it, since the schema attaches such a container to
the node of another variable.

Besides the predicate paths, the following is derived from the schema:

- **the ontology prefixes** that the queries declare, which keeps the queries from
  diverging from the semantic map that they are built for. Only the prefixes that describe
  the structure that the Triplifier produces (`dbo:`, `rdf:` and `rdfs:`) remain fixed;
- **the patient's identifier**; its predicate and class are taken from the schema's
  identifier variable.

### Schema Loading Options

The library bundles a copy of the AYA cancer schema and uses it by default. The bundled copy
is the schema of a specific release of
the [semantic map](https://github.com/STRONGAYA/AYA-cancer-semantic-map); the release that it
originates from is recorded as `BUNDLED_SCHEMA_TAG` in `schema_loader.py` and logged whenever
a schema is loaded.

```python
# Use bundled schema (default, no network needed)
df = collect_sparql_data(variables, use_schema=True)

# Or fetch the schema of the latest release (requires network access)
# Set environment variable: USE_REMOTE_SCHEMA=true

# Or pin a specific release of the semantic map
# Set environment variable: SCHEMA_TAG=v2.0.1

# Or use a custom schema URL
# Set environment variable: SCHEMA_URL=https://your-custom-url/schema.jsonld
```

Remote schemas are retrieved from a release rather than from the semantic map's main branch,
so that a node always runs against a versioned schema. A loaded schema is validated on its
structure; a document that does not describe any variables raises rather than silently
degrading every variable to `dbo:has_column`.

A weekly workflow synchronises the bundled schema with the latest release of the semantic
map and opens a pull request for review.

### Direct Schema Access

You can also use the schema loader and parser directly:

```python
from vantage6_strongaya_rdf import load_schema, get_variable_query_params

# Load the schema
schema = load_schema(use_remote=False)  # Use bundled schema

# Get query parameters for a variable
params = get_variable_query_params('age_at_initial_diagnosis', schema)
print(f"Predicate path: {params['predicate_path']}")
print(f"Main class: {params['main_class']}")
print(f"Ontology prefix: {params['ontology_prefix']}")
```

### Multi-Column Queries

The library now supports multi-column queries for fetching multiple variables in a single query:

```python
# Note: Multi-column queries fetch exactly two variables in a single query
df = collect_sparql_data(
    variables_to_extract=['variable1', 'variable2'],
    query_type="multi_column",
    endpoint="http://localhost:7200/repositories/userRepo",
    use_schema=True
)
```

Both records are identified in the same manner, after which the two attributes are kept
when their records describe the same patient:

- a record of a linked table is identified by the identifier that its foreign key
  (`dbo:fk_refers_to`) refers to;
- a record of a flat table is identified by its own identifier;
- a record whose identifier cannot be retrieved is identified by its own URI.

Note that a multi-column query only yields the patients that hold both variables, whereas a
single-column query yields the union of the patients of each variable.

A record that holds several values for the same variable - repeated measures, for instance -
is represented by one sampled value per patient, as both queries group their results by
patient.

### Patient Identifiers

The patient identifier (`patient_id`) is always text, whichever values a dataset holds; it is
never converted to a number. An identifier is a label rather than a quantity, it may hold
characters in one dataset and digits only in another, and converting it would make its type
depend on the values of a single variable - after which the variables of separate tables
could no longer be merged. Numeric identifiers are nevertheless ordered by their value rather
than as text, so `"2"` precedes `"10"`.

### Missing Patient Identifiers

A record whose identifier cannot be retrieved is identified by its own URI rather than being
dropped from the results, so that it is still observed and counted amongst the missing data.
The number of records that fell back to their URI is reported as a warning; such records
cannot be linked to the records of another table.

### Missing Values

An absence of data is an observation rather than an error: a patient that holds no value for
a requested variable remains part of the results with an empty cell, and a variable that no
record holds at all yields an empty column rather than no column.

Every notation of missing data is counted as such in the missing-value statistics that the
results carry: the Triplifier's `"NULL"` cells, a dataset's own notation (see
`MISSING_DATA_NOTATION`) and the values that a patient simply does not hold.

### Accepted Input

A variable and a variable property are substituted into the query template as they are, so
both are verified before they are used. Accepted are a class code (`ncit:C28421`), the name
of a schema variable (`age_at_initial_diagnosis`) and an IRI between angle brackets; a
property may in addition be a property path. Anything else - whitespace, braces, quotation
marks, a hash - as well as any SPARQL keyword is refused with a `UserInputError` before a
query is composed, since such input could otherwise extend the query with a clause of its
own; a federated query towards another endpoint, for instance.

### Environment Variables

The following environment variables can be used to configure schema loading:

- `USE_REMOTE_SCHEMA`: Set to `"true"` to fetch the schema from GitHub (default: `"false"`)
- `SCHEMA_TAG`: Release of the semantic map to fetch the schema from (default: its latest release)
- `SCHEMA_URL`: Custom URL to fetch schema from (overrides the release URL)
- `SPARQL_ENDPOINT`: Override the default SPARQL endpoint
- `VARIABLE_PROPERTY`: Override the default variable property predicate (only used when `use_schema=False` or as fallback)
- `MISSING_DATA_NOTATION`: Custom notation for missing data

The various functions are available through `pip install` for debugging and testing purposes.
The library can be installed as follows:

```bash
pip install git+https://github.com/STRONGAYA/v6-tools-rdf.git
```

# Testing

This repository includes a comprehensive testing framework to ensure the reliability and correctness of all functions,
especially in whether RDF-data is queryable when the library is run as a Docker container within a Vantage6 node.

## Test Structure

```
tests/
├── conftest.py                           # Common fixtures and test utilities
├── unit/                                 # Unit tests for individual functions
│   ├── test_library_functions.py         # Tests for library functions
│   ├── test_schema_functions.py          # Tests for schema loader and parser
│   ├── test_query_templates.py           # Tests for the construction of the queries
│   ├── test_query_execution.py           # Tests that execute the queries on a synthetic graph
│   ├── test_fallback_query_execution.py  # Tests that execute the queries without the schema
│   ├── test_schema_contract.py           # Tests every variable of the schema and its snapshot
│   └── snapshots/                        # Golden snapshot of every variable's predicate path
├── integration/                          # Integration tests
│   └── test_vantage6_integration.py      # Data stratification workflows
│   └── test_rdf_algorithm_integration.py # Vantage6 algorithm integration tests
├── mock_algorithm/                       # Mock Vantage6 algorithm to be used for Vantage6 integration testing
│   └── ...                               
└── data/                                 # Test data and configurations
    └── additional_vantage6_*_config.yaml # Additional Vantage6 component configurations
    └── *.ttl                             # Triplified datasets for testing
    └── rdf_store.csv                     # RDF-store reference for the Vantage6 node
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-mock hypothesis faker rdflib
```

Or install the library's test dependencies directly:

```bash
pip install -e ".[test]"
```

`rdflib` is used to execute the generated SPARQL queries against a synthetic graph, which
verifies the queries without an RDF-store; the tests that require it are skipped when it is
not installed.

### Basic Test Execution

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run specific test module
pytest tests/unit/test_library_functions.py

# Run with verbose output
pytest -v
```

### Guarding Against Schema Drift

The bundled schema is synchronised with the semantic map automatically, and it describes far
more variables than any test enumerates. Two mechanisms keep an update from silently changing
what is queried:

- `tests/unit/test_schema_contract.py` verifies every variable of the schema: it yields a
  predicate path, that path holds no transitive or optional operator, every prefix that it
  uses is declared, and the query that it produces parses as SPARQL;
- `tests/unit/snapshots/predicate_paths.json` records the predicate path, class and ontology
  prefix of every variable, so that a schema update yields a reviewable diff in the
  synchronisation pull request. Regenerate it deliberately with:

```bash
UPDATE_PREDICATE_PATH_SNAPSHOT=1 pytest tests/unit/test_schema_contract.py
```

The contract can also be verified against the latest release of the semantic map itself, which
reports an upstream change before a node encounters it. That test requires network access and
is therefore not part of the default run:

```bash
RUN_NETWORK_TESTS=1 pytest -m network
```

### Test Categories

- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test complete workflows and component interactions (whether data can be queried from the
  RDF-store in a Vantage6 node)
- **Edge Case Tests**: Test behaviour with unusual data inputs

### Test Data

The test suite uses a synthetic dataset that was triplified using
the [Triplifier](https://github.com/MaastrichtU-CDS/triplifier) tool.

### Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions:

- Multiple Python and Vantage6 versions (starting with Python 3.10 and Vantage6 4.11 and 4.12)
- Code coverage reporting
- Performance benchmarking
- Security scanning

## Contributing to Tests

When contributing new functionality:

1. **Add unit tests** for all new functions
2. **Add integration tests** for complete workflows
3. **Include edge case testing** for robustness
4. **Ensure new query templates** have corresponding tests
5. **Update test data** if needed for new scenarios; ensure that this
   is [triplified](https://github.com/MaastrichtU-CDS/triplifier).
6. **Ensure that the mock algorithm in `tests/mock_algorithm` covers the new functionality**

### Test Guidelines

- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases and scenarios
- Test edge cases and error conditions
- Use realistic synthetic data
- Validate both structure and values of results

# Contributors

- J. Hogenboom
- V. Gouthamchand

# References

- [STRONG AYA](https://strongaya.eu/)
- [Vantage6](vantage6.ai)