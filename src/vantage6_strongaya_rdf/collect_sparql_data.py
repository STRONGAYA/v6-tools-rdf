"""
------------------------------------------------------------------------------
RDF/SPARQL Data Collection Functions

File organisation:
- Query template loading (_load_query_template)
- Query processing functionalities (_process_variable_query)
- Data collection function (collect_sparql_data)
------------------------------------------------------------------------------
"""

import pandas as pd

from importlib import resources
from typing import List

# Optional import for vantage6 - only needed for vantage6 integration
try:
    from vantage6.algorithm.tools.util import get_env_var
    VANTAGE6_AVAILABLE = True
except ImportError:
    VANTAGE6_AVAILABLE = False
    # Provide fallback for get_env_var
    def get_env_var(name, default=None):
        import os
        return os.environ.get(name, default)

try:
    from vantage6_strongaya_general.miscellaneous import safe_log
    STRONGAYA_GENERAL_AVAILABLE = True
except ImportError:
    STRONGAYA_GENERAL_AVAILABLE = False
    # Provide fallback for safe_log
    def safe_log(level, message):
        print(f"[{level.upper()}] {message}")

from .sparql_client import post_sparql_query
from .data_processing import add_missing_data_info, extract_subclass_info


def _load_query_template(query_name: str) -> str:
    """
    Load the SPARQL query template from a file.

    Args:
        query_name (str): The name of the SPARQL query template file (that is located in /query_templates.

    Returns:
        str: The SPARQL query template.
    """
    try:
        with (
            resources.files("vantage6_strongaya_rdf")
            .joinpath(f"query_templates")
            .joinpath(f"{query_name}.rq")
            .open("r") as file
        ):
            return file.read()
    except Exception as e:
        safe_log("error", f"Error reading SPARQL query file: {e}.")
        return ""


def _process_variable_query(
    endpoint: str, query_template: str, variable: str, variable_property: str
) -> pd.DataFrame:
    """
    Process the SPARQL query for a single variable.

    Args:
        endpoint (str): The SPARQL endpoint URL.
        query_template (str): The SPARQL query template.
        variable (str): The variable name to query.
        variable_property (str): The property (or predicate) used to identify variables in the SPARQL query.

    Returns:
        pd.DataFrame: The DataFrame containing the query results.
    """
    ontology_part = variable.split(":")[0] + ":"
    query = (
        query_template.replace("PLACEHOLDER_CLASS", variable)
        .replace("PLACEHOLDER_ONTOLOGY", ontology_part)
        .replace("PLACEHOLDER_PREDICATE", variable_property)
    )

    safe_log("info", f"Posting SPARQL query for {variable}.")
    result = post_sparql_query(endpoint=endpoint, query=query)

    if result:
        result_df = pd.DataFrame(result)
        result_df.drop(columns=["patient"], inplace=True)
        result_df["patient_id"] = result_df.index
        return extract_subclass_info(result_df, variable)
    return pd.DataFrame(columns=[variable])


def collect_sparql_data(
    variables_to_describe: List[str],
    query_type: str = "single_column",
    endpoint: str = "http://localhost:7200/repositories/userRepo",
    variable_property: str = "dbo:has_column",
    missing_data_notation: str = "",
) -> pd.DataFrame:
    """
    Collect data from SPARQL endpoints for specified variables.

    Args:
        variables_to_describe (List[str]): List of variable names to their properties.
        query_type (str, optional): The type of query to execute. Currently, only 'single_column' is supported.
                                    Defaults to 'single_column'.
        endpoint (str, optional): The SPARQL endpoint URL.
                                  An endpoint specified in the environment variables will be prioritised.
                                  Defaults to "http://localhost:7200/repositories/userRepo".
        variable_property (str, optional): The property (or predicate) used to identify variables in the SPARQL query.
                                           A property specified in the environment variables will be prioritised.
                                           Defaults to "dbo:has_column".
        missing_data_notation (str, optional): The notation used to represent missing data in the DataFrame.
                                               A notation specified in the environment variables will be prioritised.
                                               Defaults to pd.NA.

    Returns:
        pd.DataFrame: A combined DataFrame containing all retrieved data,
        with 'patient_id' as the index column and each variable as a separate column.
    """
    # Retrieve environment variables - prioritise them over defaults as local setups might e.g. have different endpoints
    endpoint = get_env_var("SPARQL_ENDPOINT", endpoint)
    variable_property = get_env_var("VARIABLE_PROPERTY", variable_property)
    missing_data_notation = get_env_var("MISSING_DATA_NOTATION", missing_data_notation)

    if query_type == "single_column":
        query_template = _load_query_template("single_column")
    else:
        safe_log("error", f"Unknown query type: {query_type}.")
        return pd.DataFrame(columns=variables_to_describe)

    intermediate_df = pd.DataFrame(columns=["patient_id", "sub_class", "value"])

    for variable in variables_to_describe:
        try:
            result_df = _process_variable_query(
                endpoint, query_template, variable, variable_property
            )
            if not result_df.empty:
                if intermediate_df.empty:
                    intermediate_df = result_df
                else:
                    intermediate_df = pd.merge(
                        intermediate_df, result_df, on="patient_id", how="outer"
                    )
        except Exception as e:
            safe_log("error", f"Error processing {variable}: {e}")
            continue

    # add_missing_data_info(intermediate_df, missing_data_notation)
    intermediate_df = intermediate_df.replace(missing_data_notation, pd.NA)

    return (
        intermediate_df
        if not intermediate_df.empty
        else pd.DataFrame(columns=variables_to_describe)
    )
