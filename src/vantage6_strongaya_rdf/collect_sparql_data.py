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
from typing import List, Optional

from vantage6.algorithm.tools.exceptions import (
    UserInputError,
    AlgorithmError,
)
from vantage6.algorithm.tools.util import get_env_var
from vantage6_strongaya_general.miscellaneous import safe_log

from .sparql_client import post_sparql_query
from .data_processing import add_missing_data_info, extract_subclass_info, clean_null_values
from .schema_loader import load_schema
from .schema_parser import get_variable_query_params

NAUGHTY_WORD_LIST = [
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "CLEAR",
    "CREATE",
    "LOAD",
    "COPY",
    "MOVE",
    "ADD",
    "UNION",
    "SERVICE",
    "BIND",
    "FILTER",
    "OPTIONAL",
    "GRAPH",
    "CONSTRUCT",
    "DESCRIBE",
    "WITH",
    "INTO",
    "USING",
    "MINUS",
    "EXISTS",
    "NOT EXISTS",
    "FROM",
    "FROM NAMED",
    "OFFSET",
    "LIMIT",
    "ORDER BY",
    "GROUP BY",
    "HAVING",
    "DISTINCT",
    "REDUCED",
    "BINDINGS",
    "UNDEFINED",
    "LANGMATCHES",
    "DATATYPE",
    "BOUND",
    "IRI",
    "URI",
    "BNODE",
    "STR",
    "LANG",
    "ISIRI",
    "ISURI",
    "ISBLANK",
    "ISLITERAL",
    "REGEX",
    "SUBSTR",
    "REPLACE",
    "CONCAT",
    "LENGTH",
    "STRSTARTS",
    "STRENDS",
    "CONTAINS",
    "AUC",
    "COUNT",
    "SUM",
    "MIN",
    "MAX",
    "AVG",
    "SAMPLE",
    "GROUP_CONCAT",
    "SEPARATOR",
    "NOT IN",
    "IN",
    "COALESCE",
    "IF",
    "STRLANG",
    "STRDT",
    "isLiteral",
    "RAND",
    "ABS",
    "ROUND",
    "CEIL",
    "FLOOR",
]


def _load_query_template(query_name: str) -> str:
    """
    Load the SPARQL query template from a file.

    Args:
        query_name (str): The name of the SPARQL query template file (that is located in /query_templates.

    Returns:
        str: The SPARQL query template.
    """
    try:
        # Use compatible importlib.resources syntax for Python 3.8
        from importlib.resources import files, as_file
        
        # Get the package resource
        package = files("vantage6_strongaya_rdf")
        template_path = package.joinpath("query_templates").joinpath(f"{query_name}.rq")
        
        # Read the file content
        with as_file(template_path) as path:
            with open(path, "r") as file:
                return file.read()
    except Exception as e:
        safe_log("error", f"Error reading SPARQL query file: {e}.")
        return ""


def _process_variable_query(
    endpoint: str,
    query_template: str,
    variable: str,
    variable_property: str,
    schema: Optional[dict] = None,
    use_schema: bool = False,
) -> pd.DataFrame:
    """
    Process the SPARQL query for a single variable.

    Args:
        endpoint (str): The SPARQL endpoint URL.
        query_template (str): The SPARQL query template.
        variable (str): The variable name to query.
        variable_property (str): The property (or predicate) used to identify variables in the SPARQL query.
        schema (Optional[dict]): The JSON-LD schema dictionary (if use_schema is True).
        use_schema (bool): Whether to use schema-based predicate path generation.

    Returns:
        pd.DataFrame: The DataFrame containing the query results.
    """
    # Check for naughty words in the input variables early
    ontology_part = variable.split(":")[0] + ":"
    if any(
        word in (variable, ontology_part, variable_property)
        for word in NAUGHTY_WORD_LIST
    ):
        raise UserInputError(
            "Potentially dangerous input detected in variable, ontology part, or variable property."
        )

    # Build query based on whether we're using schema or not
    if use_schema and schema:
        # Use schema-based predicate path generation
        query_params = get_variable_query_params(variable, schema)

        if not query_params:
            safe_log(
                "warning",
                f"Could not get query params for {variable} from schema, using fallback",
            )
            # Fallback to simple replacement
            predicate_path = variable_property
            main_class = variable
            ontology_prefix = ontology_part
        else:
            predicate_path = query_params.get("predicate_path", variable_property)
            main_class = query_params.get("main_class", variable)
            ontology_prefix = query_params.get("ontology_prefix", ontology_part)

        query = (
            query_template.replace("PLACEHOLDER_CLASS", main_class)
            .replace("PLACEHOLDER_ONTOLOGY", ontology_prefix)
            .replace("PLACEHOLDER_PREDICATE_PATH", predicate_path)
            .replace("PLACEHOLDER_PREDICATE", predicate_path)  # Backward compatibility
        )
    else:
        # Use simple placeholder replacement (backward compatible)
        query = (
            query_template.replace("PLACEHOLDER_CLASS", variable)
            .replace("PLACEHOLDER_ONTOLOGY", ontology_part)
            .replace("PLACEHOLDER_PREDICATE_PATH", variable_property)
            .replace("PLACEHOLDER_PREDICATE", variable_property)
        )

    safe_log("info", f"Posting SPARQL query for {variable}.")
    result = post_sparql_query(endpoint=endpoint, query=query)

    if result:
        result_df = pd.DataFrame(result)

        # Handle both old and new column names
        if "patient" in result_df.columns:
            result_df.drop(columns=["patient"], inplace=True)

        # Use patientID if available, otherwise use index
        if "patientID" in result_df.columns:
            result_df["patient_id"] = result_df["patientID"]
            result_df.drop(columns=["patientID"], inplace=True)
            # Convert patient_id to numeric if possible for proper sorting
            try:
                result_df["patient_id"] = pd.to_numeric(result_df["patient_id"])
            except (ValueError, TypeError):
                pass  # Keep as string if conversion fails
        else:
            result_df["patient_id"] = result_df.index

        # Handle subClass column name variations
        if "subClass" in result_df.columns:
            result_df.rename(columns={"subClass": "sub_class"}, inplace=True)

        return clean_null_values(extract_subclass_info(result_df, variable))
    else:
        return pd.DataFrame(columns=["patient_id", variable])


def _process_multi_column_query(
    endpoint: str,
    query_template: str,
    variables: List[str],
    variable_property: str,
    schema: Optional[dict] = None,
    use_schema: bool = False,
) -> pd.DataFrame:
    """
    Process the multi-column SPARQL query for two variables in a single query.

    The multi_column template fetches both variables together, returning combined
    results with subClass/any_value for the first variable and subClass2/any_value2
    for the second.

    Args:
        endpoint (str): The SPARQL endpoint URL.
        query_template (str): The multi-column SPARQL query template.
        variables (List[str]): The two variable names to query.
        variable_property (str): Fallback predicate property.
        schema (Optional[dict]): The JSON-LD schema dictionary.
        use_schema (bool): Whether to use schema-based predicate path generation.

    Returns:
        pd.DataFrame: The DataFrame containing the combined query results.
    """
    if len(variables) != 2:
        raise UserInputError(
            f"Multi-column query requires exactly 2 variables, but {len(variables)} were provided."
        )

    var1, var2 = variables

    # Check for naughty words in both variables
    for variable in [var1, var2]:
        ontology_part = variable.split(":")[0] + ":"
        if any(
            word in (variable, ontology_part, variable_property)
            for word in NAUGHTY_WORD_LIST
        ):
            raise UserInputError(
                "Potentially dangerous input detected in variable, ontology part, "
                "or variable property."
            )

    # Build query parameters for each variable
    query = query_template
    for idx, variable in enumerate([var1, var2], start=1):
        suffix = f"_{idx}"
        ontology_part = variable.split(":")[0] + ":"

        if use_schema and schema:
            query_params = get_variable_query_params(variable, schema)
            if query_params:
                predicate_path = query_params.get("predicate_path", variable_property)
                main_class = query_params.get("main_class", variable)
                ontology_prefix = query_params.get("ontology_prefix", ontology_part)
            else:
                safe_log(
                    "warning",
                    f"Could not get query params for {variable} from schema, "
                    "using fallback",
                )
                predicate_path = variable_property
                main_class = variable
                ontology_prefix = ontology_part
        else:
            predicate_path = variable_property
            main_class = variable
            ontology_prefix = ontology_part

        query = (
            query.replace(f"PLACEHOLDER_CLASS{suffix}", main_class)
            .replace(f"PLACEHOLDER_ONTOLOGY{suffix}", ontology_prefix)
            .replace(f"PLACEHOLDER_PREDICATE_PATH{suffix}", predicate_path)
        )

    safe_log("info", f"Posting multi-column SPARQL query for {var1} and {var2}.")
    result = post_sparql_query(endpoint=endpoint, query=query)

    if not result:
        return pd.DataFrame(columns=["patient_id", var1, var2])

    result_df = pd.DataFrame(result)

    # Drop patient URI columns
    for col in ["p1", "p2"]:
        if col in result_df.columns:
            result_df.drop(columns=[col], inplace=True)

    # Use patientID if available, otherwise use index
    if "patientID" in result_df.columns:
        result_df["patient_id"] = result_df["patientID"]
        result_df.drop(columns=["patientID"], inplace=True)
        try:
            result_df["patient_id"] = pd.to_numeric(result_df["patient_id"])
        except (ValueError, TypeError):
            pass
    else:
        result_df["patient_id"] = result_df.index

    # Process first variable: subClass + any_value -> var1
    if "subClass" in result_df.columns:
        result_df.rename(columns={"subClass": "sub_class"}, inplace=True)
    result_df = extract_subclass_info(result_df, var1)

    # Process second variable: subClass2 + any_value2 -> var2
    if "subClass2" in result_df.columns and "any_value2" in result_df.columns:
        result_df[var2] = result_df.apply(
            lambda row: (
                row["any_value2"]
                if pd.isna(row.get("subClass2")) or row.get("subClass2") == ""
                else row["subClass2"]
            ),
            axis=1,
        )
        cols_to_drop = ["subClass2", "any_value2"]
        result_df.drop(
            columns=[c for c in cols_to_drop if c in result_df.columns],
            inplace=True,
        )
    elif "any_value2" in result_df.columns:
        result_df.rename(columns={"any_value2": var2}, inplace=True)
        if "subClass2" in result_df.columns:
            result_df.drop(columns=["subClass2"], inplace=True)

    # Clean NULL values to handle string representations like "['NULL']"
    result_df = clean_null_values(result_df)

    return result_df


def collect_sparql_data(
    variables_to_describe: List[str],
    query_type: str = "single_column",
    endpoint: str = "http://localhost:7200/repositories/userRepo",
    variable_property: Optional[str] = None,
    missing_data_notation: str = "",
    use_schema: bool = False,
    schema_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect data from SPARQL endpoints for specified variables.

    Args:
        variables_to_describe (List[str]): List of variable names to their properties.
        query_type (str, optional): The type of query to execute. Supports 'single_column' and 'multi_column'.
                                    Defaults to 'single_column'.
        endpoint (str, optional): The SPARQL endpoint URL.
                                  An endpoint specified in the environment variables will be prioritised.
                                  Defaults to "http://localhost:7200/repositories/userRepo".
        variable_property (str, optional): The property (or predicate) used to identify variables in the
                                           SPARQL query. A property specified in the environment variables will be
                                           prioritised. Only required when use_schema is False. Defaults to
                                           "dbo:has_column" if not provided.
        missing_data_notation (str, optional): The notation used to represent missing data in the DataFrame.
                                               A notation specified in the environment variables will be prioritised.
                                               Defaults to pd.NA.
        use_schema (bool, optional): Whether to use schema-based predicate path generation.
                                     Defaults to False for backward compatibility.
        schema_url (str, optional): Custom URL to fetch schema from. Only used if use_schema is True.

    Returns:
        pd.DataFrame: A combined DataFrame containing all retrieved data,
        with 'patient_id' as the index column and each variable as a separate column.
    """
    # Retrieve environment variables - prioritise them over defaults as local setups might e.g. have different endpoints
    endpoint = get_env_var("SPARQL_ENDPOINT", endpoint)

    # Set default for variable_property if not provided
    if variable_property is None:
        variable_property = "dbo:has_column"

    variable_property = get_env_var("VARIABLE_PROPERTY", variable_property)
    missing_data_notation = get_env_var("MISSING_DATA_NOTATION", missing_data_notation)

    # Load schema if needed
    schema = None
    if use_schema:
        try:
            # Check if we should use remote schema
            use_remote = get_env_var("USE_REMOTE_SCHEMA", "false").lower() == "true"
            schema_url_env = get_env_var("SCHEMA_URL", schema_url)

            schema = load_schema(
                use_remote=use_remote, schema_url=schema_url_env, local_fallback=True
            )
        except Exception as e:
            safe_log("error", f"Failed to load schema: {e}")
            raise AlgorithmError("error", f"Failed to load schema: {e}")

    if query_type == "single_column":
        query_template = _load_query_template("single_column")

        intermediate_df = pd.DataFrame(columns=["patient_id", "sub_class", "value"])

        for variable in variables_to_describe:
            try:
                result_df = _process_variable_query(
                    endpoint,
                    query_template,
                    variable,
                    variable_property,
                    schema,
                    use_schema,
                )
                if not result_df.empty:
                    if intermediate_df.empty:
                        intermediate_df = result_df
                    else:
                        intermediate_df = pd.merge(
                            intermediate_df,
                            result_df,
                            on="patient_id",
                            how="outer",
                        )
            except Exception as e:
                raise AlgorithmError("error", f"Error processing {variable}: {e}")

    elif query_type == "multi_column":
        query_template = _load_query_template("multi_column")

        try:
            intermediate_df = _process_multi_column_query(
                endpoint,
                query_template,
                variables_to_describe,
                variable_property,
                schema,
                use_schema,
            )
        except Exception as e:
            raise AlgorithmError(
                "error",
                f"Error processing multi-column query: {e}",
            )

    else:
        raise UserInputError(f"Unknown query type: {query_type}.")

    # Calculate the missing count using the specific notation
    add_missing_data_info(intermediate_df, missing_data_notation)

    # Replace the missing value notation to prevent TypeErrors
    intermediate_df = intermediate_df.replace(missing_data_notation, pd.NA)

    # Sort by patient_id to ensure consistent ordering
    if not intermediate_df.empty and "patient_id" in intermediate_df.columns:
        intermediate_df = intermediate_df.sort_values("patient_id").reset_index(
            drop=True
        )

    return intermediate_df
