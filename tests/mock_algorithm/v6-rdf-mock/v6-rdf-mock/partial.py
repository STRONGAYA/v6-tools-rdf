import pandas as pd
from typing import Any, Dict

from vantage6.algorithm.tools.decorators import data

from vantage6_strongaya_general.miscellaneous import safe_log, set_datatypes, CategoricalDetails, NonCategoricalDetails
from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data


@data(1)
def partial_rdf_mock(
    df: pd.DataFrame, variables_to_extract: Dict[str, CategoricalDetails | NonCategoricalDetails], query_type: str
) -> Any:
    """
    Decentral part of the algorithm

    Args:
        df (pd.DataFrame): The input DataFrame containing at least the 'endpoint' column.
        variables_to_extract (Dict[str, str]): Dict of variables to extract from the RDF database.
        query_type (str): The type of SPARQL query to use (default is "single_column").

    Returns:
        Any: The result of the SPARQL query in JSON format.
    """

    safe_log("info", "Starting partial algorithm function")

    # Extract the list of variable to extract
    list_of_variables = list(variables_to_extract.keys())

    result = collect_sparql_data(
        list_of_variables,
        query_type=query_type,
        endpoint=df["endpoint"].iloc[0],
    )

    # Set the datatype to ensure conversions can be done properly
    result = set_datatypes(result, variables_to_extract)

    safe_log("info", "Returning queried RDF-database contents")
    return result.to_json()
