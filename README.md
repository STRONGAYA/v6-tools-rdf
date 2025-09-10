# STRONG AYA's RDF Vantage6 tools
<p align="center">
<a href="https://github.com/STRONGAYA/v6-tools-general/workflows/"><img alt="Test status" src="https://github.com/STRONGAYA/v6-tools-general/workflows/Test%20Suite/badge.svg"></a>
<a><img alt="Coverage" src="https://raw.githubusercontent.com/STRONGAYA/v6-tools-rdf/main/tests/coverage-badge.svg"></a>
<a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img alt="Licence: Apache 2.0" src="https://img.shields.io/badge/Licence-Apache%202.0-blue.svg"></a>
<br>
<a href="https://github.com/vantage6/vantage6/"><img alt="Vantage6 4.11, 4.12" src="https://img.shields.io/badge/vantage6- 4.11 | 4.12-blue.svg"></a>
<a href="https://github.com/MaastrichtU-CDS/Flyover"><img alt="Flyover version 2.0+" src="https://img.shields.io/badge/Flyover%20Version-2.0+-purple"></a>
<a href="https://strongaya.eu/wp-content/uploads/2025/07/algorithm_review_guidelines.pdf"><img alt="STRONG AYA Algorithm Guideline Conformity: v1.0.0 Pending" src="https://img.shields.io/badge/STRONG%20AYA%20Algorithm%20Guideline%20Conformity-v1.0.0%20pending-yellow"></a>
<br>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://flake8.pycqa.org/"><img alt="Linting: flake8" src="https://img.shields.io/badge/linting-flake8-informational"></a>
<a href="http://mypy-lang.org/"><img alt="Type checking: mypy" src="https://img.shields.io/badge/type%20checking-mypy-informational"></a>
<a href="https://github.com/PyCQA/bandit"><img alt="Security: bandit" src="https://img.shields.io/badge/security-bandit-informational"></a>
<a href="https://github.com/pyupio/safety"><img alt="Security: safety" src="https://img.shields.io/badge/security-safety-informational"></a>
</p>

<!--
To show the approved badge instead, use:
<a href="https://strongaya.eu/wp-content/uploads/2025/07/algorithm_review_guidelines.pdf"><img alt="STRONG AYA Algorithm Guideline Conformity: v1.0.0 Approved" src="https://img.shields.io/badge/STRONG%20AYA%20Algorithm%20Guideline%20Conformity-v1.0.0%20approved-brightgreen">
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
- **Query Templates**: SPARQL query templates that the SPARQL data collection section uses

# Usage

The library provides functions that can be included in a Vantage6 algorithm as the algorithm developer sees fit.
The functions are designed to be modular and can be used independently or in combination with other functions.

The library can be included in your Vantage6 algorithm by listing it in the `requirements.txt` and `setup.py` file of
your
algorithm.

## Including the library in your Vantage6 algorithm

For the `requirements.txt` file, you can add the following line to the file:

```
git+https://github.com/STRONGAYA/v6-tools-rdf.git@v0.1.3
```

For the `setup.py` file, you can add the following line to the `install_requires` list:

```python
        "vantage6-strongaya-rdf @ git+https://github.com/STRONGAYA/v6-tools-rdf.git@v0.1.3",
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
    version="1.0.0",
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
        "vantage6-strongaya-rdf @ git+https://github.com/STRONGAYA/v6-tools-rdf.git@v0.1.3"
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


def partial_general_statistics(variables_to_analyse: dict) -> dict:
    """
    Execute the partial algorithm for some modelling using RDF data.

    Args:
        variables_to_analyse (list): List of variables to analyse.

    Returns:
        dict: A dictionary containing the computed general statistics.
    """
    safe_log("info", "Executing partial algorithm for some modelling using RDF data.")

    # Set datatypes for each variable
    df = collect_sparql_data(variables_to_analyse, query_type="single_column",
                             endpoint="http://localhost:7200/repositories/userRepo",
                             )

    # Ensure that the desired privacy measures are applied

    # Do some modelling of the data

    return result
```

The various functions are available through `pip install` for debugging and testing purposes.
The library can be installed as follows:

```bash
pip install git+https://github.com/STRONGAYA/v6-tools-rdf.git
```

# Contributers

- J. Hogenboom
- V. Gouthamchand

# References

- [STRONG AYA](https://strongaya.eu/)
- [Vantage6](vantage6.ai)