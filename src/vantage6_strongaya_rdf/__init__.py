"""
vantage6-strongaya-rdf - A library with RDF functions for Vantage6 algorithms
"""

from .collect_sparql_data import collect_sparql_data
from .schema_loader import load_schema
from .schema_parser import (
    build_predicate_path,
    get_variable_query_params,
    resolve_intermediate_class_path,
)

__all__ = [
    "collect_sparql_data",
    "load_schema",
    "build_predicate_path",
    "get_variable_query_params",
    "resolve_intermediate_class_path",
]

__version__ = "1.0.1"
