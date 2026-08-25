"""
vantage6-strongaya-rdf - A library with RDF functions for Vantage6 algorithms
"""

from .collect_sparql_data import collect_sparql_data
from .schema_loader import (
    get_schema_version,
    load_schema,
    resolve_latest_schema_tag,
    validate_schema,
)
from .schema_parser import (
    build_predicate_path,
    get_identifier_query_params,
    get_intermediate_classes,
    get_schema_prefixes,
    get_variable_instance_path,
    get_variable_query_params,
    resolve_intermediate_class_path,
)

__all__ = [
    "collect_sparql_data",
    "load_schema",
    "get_schema_version",
    "resolve_latest_schema_tag",
    "validate_schema",
    "build_predicate_path",
    "get_identifier_query_params",
    "get_intermediate_classes",
    "get_schema_prefixes",
    "get_variable_instance_path",
    "get_variable_query_params",
    "resolve_intermediate_class_path",
]

__version__ = "1.1.1"
