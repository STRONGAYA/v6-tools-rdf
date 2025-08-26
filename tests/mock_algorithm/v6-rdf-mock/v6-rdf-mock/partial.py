"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import pandas as pd
from typing import Any, List

from vantage6.algorithm.tools.decorators import data

from vantage6_strongaya_general.miscellaneous import safe_log
from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data

# TODO add British english Docstring and update any existing info logging; keep it concise
@data(1)
def partial_rdf_mock(
    df: pd.DataFrame, variables_to_extract: List[str]
) -> Any:
    """ Decentral part of the algorithm """
    safe_log("info", "Starting partial algorithm function")

    result = collect_sparql_data(variables_to_extract, query_type="single_column",
                             endpoint=df["endpoint"].iloc[0],
                             )

    safe_log("info", "Returning queried RDF-database contents")
    return result.to_json()

