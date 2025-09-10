import pandas as pd
from typing import Any, List

from vantage6.algorithm.tools.decorators import data

from vantage6_strongaya_general.miscellaneous import safe_log
from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data


@data(1)
def partial_rdf_mock(df: pd.DataFrame, variables_to_extract: List[str]) -> Any:
    """
    Decentral part of the algorithm

    Args:
        df (pd.DataFrame): The input DataFrame containing at least the 'endpoint' column.
        variables_to_extract (List[str]): List of variables to extract from the RDF database.
    """

    safe_log("info", "Starting partial algorithm function")

    result = collect_sparql_data(
        variables_to_extract,
        query_type="single_column",
        endpoint=df["endpoint"].iloc[0],
    )

    safe_log("info", "Returning queried RDF-database contents")
    return result.to_json()
