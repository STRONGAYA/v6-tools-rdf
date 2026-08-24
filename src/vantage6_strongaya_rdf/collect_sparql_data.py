"""
------------------------------------------------------------------------------
RDF/SPARQL Data Collection Functions

File organisation:
- Query template loading (_load_query_template)
- Concurrent query execution (_execute_concurrently)
- Query processing functionalities (_process_variable_query)
- Data collection function (collect_sparql_data)
------------------------------------------------------------------------------
"""

import pandas as pd
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from importlib import resources
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

from vantage6.algorithm.tools.exceptions import (
    UserInputError,
    AlgorithmError,
)
from vantage6.algorithm.tools.util import get_env_var
from vantage6_strongaya_general.miscellaneous import safe_log

from .sparql_client import post_sparql_query
from .data_processing import (
    add_missing_data_info,
    extract_subclass_info,
    clean_null_values,
)
from .schema_loader import get_schema_version, load_schema
from .schema_parser import (
    DEFAULT_IDENTIFIER_CLASS,
    DEFAULT_IDENTIFIER_PREDICATE,
    get_identifier_query_params,
    get_schema_prefixes,
    get_variable_instance_path,
    get_variable_query_params,
)

# Prefixes of the structure that the Triplifier produces; these describe the database
# ontology rather than the semantic map and are therefore not part of the schema
STRUCTURAL_PREFIXES = {
    "dbo": "http://um-cds/ontologies/databaseontology/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}

# Ontology prefixes that are used when no schema is available to derive them from
DEFAULT_ONTOLOGY_PREFIXES = {
    "ncit": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
    "roo": "http://www.cancerdata.org/roo/",
    "sct": "http://snomed.info/id/",
    "sio": "http://semanticscience.org/resource/",
    "strongaya": "https://strongaya.eu/",
}

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


# Shapes that a variable and a variable property may take. A variable is either a class
# code ("ncit:C28421"), the name of a schema variable ("age_at_initial_diagnosis") or an
# IRI between angle brackets; a property may in addition be a property path. Any other
# shape - and in particular one that holds whitespace, braces, quotation marks or a hash -
# is refused, as it would allow the input to close the triple pattern that it is
# substituted into and to append a clause of its own to the query.
PERMITTED_IRI_PATTERN = r"<[^\s<>{}\"'`|\\^]+>"
PERMITTED_VARIABLE_PATTERN = re.compile(
    r"^(?:[A-Za-z_][\w.\-]*(?::[\w.\-]+)?|" + PERMITTED_IRI_PATTERN + r")$"
)
PERMITTED_PROPERTY_PATTERN = re.compile(
    r"^(?:[A-Za-z_(][\w.\-:/|()^]*|" + PERMITTED_IRI_PATTERN + r")$"
)


def _verify_input_safety(variable: str, variable_property: str) -> None:
    """
    Verify that a variable and a property can be substituted into a query safely.

    Both are substituted into the query template as they are, which means that an input
    of an unexpected shape could extend the query with a clause of its own; a federated
    query towards another endpoint, for instance. The shape of the input is therefore
    verified before it is used, and any SPARQL keyword within it is refused as well.

    Args:
        variable (str): The variable (or class code) to verify.
        variable_property (str): The property that identifies the variable.

    Raises:
        UserInputError: If either input does not hold the expected shape or holds a
                        SPARQL keyword.
    """
    for description, value, pattern in (
        ("variable", variable, PERMITTED_VARIABLE_PATTERN),
        ("variable property", variable_property, PERMITTED_PROPERTY_PATTERN),
    ):
        if not isinstance(value, str) or not pattern.match(value):
            raise UserInputError(
                f"Potentially dangerous input detected in the {description}; only a "
                f"class code, a variable name or an IRI between angle brackets is "
                f"accepted."
            )

    # The keywords are sought as whole words, as they occur within perfectly ordinary
    # names as well; "IN" within "initial", for instance
    for candidate in (variable, variable.split(":")[0] + ":", variable_property):
        for word in NAUGHTY_WORD_LIST:
            if re.search(r"\b" + re.escape(word) + r"\b", candidate, re.IGNORECASE):
                raise UserInputError(
                    "Potentially dangerous input detected in variable, ontology part, "
                    f"or variable property; '{word}' is a SPARQL keyword."
                )


def _natural_sort_key(identifier: object) -> Tuple[Union[str, int], ...]:
    """
    Compose a sort key that orders textual identifiers in their natural order.

    Identifiers are text, which means that a plain sort would place "10" before "2".
    The numeric parts of an identifier are therefore compared as numbers, whereas its
    textual parts are compared as text. The key always alternates between text and
    number, so that identifiers of any shape remain comparable to one another.

    Args:
        identifier (object): The identifier to compose a sort key for.

    Returns:
        Tuple[Union[str, int], ...]: The identifier's sort key.
    """
    if identifier is None or identifier is pd.NA:
        return ("",)

    parts = re.split(r"(\d+)", str(identifier))

    return tuple(int(part) if index % 2 else part for index, part in enumerate(parts))


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
            .joinpath("query_templates")
            .joinpath(f"{query_name}.rq")
            .open("r") as file
        ):
            return file.read()
    except Exception as e:
        safe_log("error", f"Error reading SPARQL query file: {e}.")
        return ""


# The type of the result that a task yields; kept generic so that the helper below can
# be reused by any strategy that processes a list of independent items, rather than
# being tied to the DataFrame that the single-column query happens to produce
_TaskResult = TypeVar("_TaskResult")


def _execute_concurrently(
    tasks: Sequence[Tuple[str, Callable[[], _TaskResult]]],
    max_concurrency: int,
) -> Iterator[Tuple[str, Optional[_TaskResult], Optional[Exception]]]:
    """
    Execute a list of labelled tasks, yielding each result as soon as it completes.

    A concurrency of 1 (or less) runs the tasks one after another, in the order that
    they were given, which keeps the default behaviour unchanged; a higher concurrency
    runs them in a thread pool instead, which is worthwhile since each task merely
    waits on an HTTP response rather than doing CPU-bound work. Any exception that a
    task raises is caught and yielded alongside its label rather than propagated, so
    that the failure of one task does not keep the results of the others from being
    processed.

    Args:
        tasks (Sequence[Tuple[str, Callable[[], _TaskResult]]]): The tasks to execute,
                                                                  each given as a label
                                                                  (used for logging and
                                                                  for reporting which
                                                                  task a result or
                                                                  failure belongs to)
                                                                  and the callable that
                                                                  performs the task
                                                                  itself.
        max_concurrency (int): The maximum number of tasks that may run at the same
                               time.

    Yields:
        Tuple[str, Optional[_TaskResult], Optional[Exception]]: The label of the task,
        its result (or None if it failed) and the exception that it raised (or None if
        it succeeded).
    """
    if max_concurrency <= 1 or len(tasks) <= 1:
        for label, task in tasks:
            try:
                yield label, task(), None
            except Exception as e:
                yield label, None, e
        return

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_label = {executor.submit(task): label for label, task in tasks}
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                yield label, future.result(), None
            except Exception as e:
                yield label, None, e


def _build_prefix_declarations(schema: Optional[dict] = None) -> str:
    """
    Compose the PREFIX declarations of a SPARQL query.

    The ontology prefixes are derived from the schema, which prevents the queries from
    diverging from the semantic map that they are built for. The prefixes that describe
    the Triplifier's structure are always declared, as the schema does not hold them.

    Args:
        schema (Optional[dict]): The JSON-LD schema dictionary; when it is not provided,
                                 the default ontology prefixes are declared instead.

    Returns:
        str: The PREFIX declarations of the query.
    """
    prefixes = dict(DEFAULT_ONTOLOGY_PREFIXES)

    if schema:
        schema_prefixes = get_schema_prefixes(schema)
        if schema_prefixes:
            # The schema is authoritative for the ontologies that it describes
            prefixes.update(schema_prefixes)
        else:
            safe_log(
                "warn",
                "Schema does not declare any prefixes; using the default prefixes",
            )

    # The Triplifier's structure is not described by the schema and cannot be overridden
    prefixes.update(STRUCTURAL_PREFIXES)

    return "\n".join(
        f"PREFIX {prefix}: <{uri}>" for prefix, uri in sorted(prefixes.items())
    )


def _prepare_query_template(query_template: str, schema: Optional[dict] = None) -> str:
    """
    Complete the parts of a query template that do not depend on the queried variables.

    Args:
        query_template (str): The SPARQL query template.
        schema (Optional[dict]): The JSON-LD schema dictionary; when it is not provided,
                                 the default prefixes and identifier are used instead.

    Returns:
        str: The query template with its prefixes and identifier filled in.
    """
    if schema:
        identifier_params = get_identifier_query_params(schema)
    else:
        identifier_params = {
            "predicate": DEFAULT_IDENTIFIER_PREDICATE,
            "class": DEFAULT_IDENTIFIER_CLASS,
        }

    return (
        query_template.replace(
            "PLACEHOLDER_PREFIXES", _build_prefix_declarations(schema)
        )
        .replace("PLACEHOLDER_ID_PREDICATE", identifier_params["predicate"])
        .replace("PLACEHOLDER_ID_CLASS", identifier_params["class"])
    )


def _assign_patient_id(result_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Determine the patient identifier of a query result.

    A record whose identifier could not be retrieved is identified by its own URI, so
    that the record is still observed and counted rather than silently disappearing
    from the results. Such records are reported, as they cannot be linked to the
    records of another table.

    Args:
        result_df (pd.DataFrame): The DataFrame containing the query results.
        variable (str): The variable(s) that were queried; used for reporting.

    Returns:
        pd.DataFrame: The DataFrame with a 'patient_id' column.
    """
    if "patientID" not in result_df.columns:
        raise AlgorithmError(
            f"Query results for {variable} do not hold a patient identifier."
        )

    # Columns that hold the URI of the record that the values originate from
    record_columns = [
        column for column in ["patient", "p1", "p2"] if column in result_df.columns
    ]

    identifiers = result_df["patientID"].astype(str)
    for record_column in record_columns:
        fallback_count = int(
            (identifiers == result_df[record_column].astype(str)).sum()
        )
        if fallback_count:
            safe_log(
                "warn",
                f"Could not retrieve the identifier of {fallback_count} record(s) while "
                f"querying {variable}; the record's own URI is used as identifier "
                f"instead, which means that these records cannot be linked to the "
                f"records of another table.",
            )

    # Identifiers are kept as text, as that is what an RDF-store returns and what a
    # dataset's identifiers often are. Converting them to a number where they happen to
    # be numeric would make the identifier's type depend on the values of a single
    # variable, after which the variables of separate tables could no longer be merged.
    result_df["patient_id"] = identifiers
    result_df = result_df.drop(columns=["patientID"] + record_columns)

    return result_df


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
    # Verify the input before it is substituted into the query template
    _verify_input_safety(variable, variable_property)

    ontology_part = variable.split(":")[0] + ":"

    # Build query based on whether we're using schema or not
    if use_schema and schema:
        # Use schema-based predicate path generation
        query_params = get_variable_query_params(variable, schema)

        if not query_params:
            safe_log(
                "warn",
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
        )
    else:
        # Use simple placeholder replacement (backward compatible)
        query = (
            query_template.replace("PLACEHOLDER_CLASS", variable)
            .replace("PLACEHOLDER_ONTOLOGY", ontology_part)
            .replace("PLACEHOLDER_PREDICATE_PATH", variable_property)
        )

    safe_log("info", f"Posting SPARQL query for {variable}.")
    result = post_sparql_query(
        endpoint=endpoint, query=query, log_label=f"Query for {variable}"
    )

    if result:
        safe_log(
            "info", f"Query for {variable} completed and returned {len(result)} row(s)."
        )
        result_df = _assign_patient_id(pd.DataFrame(result), variable)

        # Handle subClass column name variations
        if "subClass" in result_df.columns:
            result_df.rename(columns={"subClass": "sub_class"}, inplace=True)

        return clean_null_values(extract_subclass_info(result_df, variable))
    else:
        safe_log("info", f"Query for {variable} completed and returned no rows.")
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

    When one variable is filled in as part of a PROM, EHR or HCPROM container and the
    other is itself one of that container's own entries (see get_variable_instance_path),
    the two are additionally correlated on the container that they share, so that an
    attribute is not paired with the recording of a different administration of the
    same patient.

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

    # Verify both variables before either is substituted into the query template
    for variable in [var1, var2]:
        _verify_input_safety(variable, variable_property)

    # Build query parameters for each variable
    query = query_template
    predicate_paths = {}
    instance_paths = {}
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
                    "warn",
                    f"Could not get query params for {variable} from schema, "
                    "using fallback",
                )
                predicate_path = variable_property
                main_class = variable
                ontology_prefix = ontology_part
            instance_paths[idx] = get_variable_instance_path(variable, schema)
        else:
            predicate_path = variable_property
            main_class = variable
            ontology_prefix = ontology_part
            instance_paths[idx] = {}

        predicate_paths[idx] = predicate_path

        query = query.replace(f"PLACEHOLDER_CLASS{suffix}", main_class).replace(
            f"PLACEHOLDER_ONTOLOGY{suffix}", ontology_prefix
        )

    # The two attributes are only correlated on the container that they were recorded
    # within - a questionnaire answer and its own recording timestamp, for instance -
    # when one of them is recorded as part of the container ("after") and the other as
    # one of the container's own entries ("before"), and both refer to the very same
    # container class. Correlating two "after" occurrences of the same container class
    # is deliberately excluded, as that would require two ordinary attributes that
    # merely reference the same kind of container (e.g. an EHR entry) to originate from
    # that exact same entry, which is not the intent of this correlation.
    roles = {idx: info.get("role") for idx, info in instance_paths.items()}
    correlate_on_instance = (
        roles[1]
        and roles[2]
        and not (roles[1] == "after" and roles[2] == "after")
        and instance_paths[1].get("instance_class")
        == instance_paths[2].get("instance_class")
    )
    if not correlate_on_instance:
        instance_paths = {1: {}, 2: {}}

    for idx in (1, 2):
        suffix = f"_{idx}"
        attr_variable = "attr" if idx == 1 else "attr2"
        instance_info = instance_paths[idx]

        if instance_info.get("role") == "before":
            fetch_block = (
                f"?p{idx} {instance_info['path_to_instance']} ?instance{idx} .\n"
                f"  ?instance{idx} {instance_info['hop_to_value']} ?{attr_variable} ."
            )
            instance_block = ""
        elif instance_info.get("role") == "after":
            fetch_block = f"?p{idx} {predicate_paths[idx]} ?{attr_variable} ."
            instance_block = (
                f"OPTIONAL {{ ?{attr_variable} {instance_info['hop_predicate']} "
                f"?instance{idx} . }}"
            )
        else:
            fetch_block = f"?p{idx} {predicate_paths[idx]} ?{attr_variable} ."
            instance_block = ""

        query = query.replace(f"PLACEHOLDER_FETCH_BLOCK{suffix}", fetch_block).replace(
            f"PLACEHOLDER_INSTANCE_BLOCK{suffix}", instance_block
        )

    safe_log("info", f"Posting multi-column SPARQL query for {var1} and {var2}.")
    result = post_sparql_query(
        endpoint=endpoint,
        query=query,
        log_label=f"Multi-column query for {var1} and {var2}",
    )

    if not result:
        safe_log(
            "info",
            f"Multi-column query for {var1} and {var2} completed and returned no rows.",
        )
        return pd.DataFrame(columns=["patient_id", var1, var2])

    safe_log(
        "info",
        f"Multi-column query for {var1} and {var2} completed and returned "
        f"{len(result)} row(s).",
    )

    result_df = _assign_patient_id(pd.DataFrame(result), f"{var1} and {var2}")

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
    variables_to_extract: List[str],
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
        variables_to_extract (List[str]): List of variables to extract; either the schema's
                                          variable names or their class codes.
        query_type (str, optional): The type of query to execute. Supports 'single_column' and 'multi_column'.
                                    Defaults to 'single_column'. The queries of a 'single_column'
                                    extraction may be run concurrently; see SPARQL_MAX_CONCURRENCY below.
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
                                    A URL specified in the environment variables will be prioritised,
                                    as will a release tag specified through SCHEMA_TAG.

    Returns:
        pd.DataFrame: A combined DataFrame containing all retrieved data,
        with 'patient_id' as the index column and each variable as a separate column.

    Note:
        A handful of environment variables tune how a query is posted, rather than
        being arguments of this function, as they concern the endpoint's reliability
        rather than the data that is requested: SPARQL_TIMEOUT (the number of seconds
        that a request may take, 60 by default), SPARQL_MAX_RETRIES (the number of
        times a failed request is retried, 3 by default) and SPARQL_MAX_CONCURRENCY
        (the number of 'single_column' queries that may be posted at once, 1 - i.e.
        sequential - by default). A variable whose query keeps failing after these
        retries are exhausted is skipped, and reported as such, so that it does not
        keep the rest of a 'single_column' extraction from being collected.
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
            schema_tag_env = get_env_var("SCHEMA_TAG", None)

            schema = load_schema(
                use_remote=use_remote,
                schema_url=schema_url_env,
                schema_tag=schema_tag_env,
                local_fallback=True,
            )
            safe_log(
                "info",
                f"Using AYA cancer schema version {get_schema_version(schema)}",
            )
        except Exception as e:
            safe_log("error", f"Failed to load schema: {e}")
            raise AlgorithmError(f"Failed to load schema: {e}")

    if query_type == "single_column":
        query_template = _prepare_query_template(
            _load_query_template("single_column"), schema
        )

        intermediate_df = pd.DataFrame(columns=["patient_id", "sub_class", "value"])

        # A triplestore may accept several concurrent queries and simply queue up any
        # that exceed its own capacity, which is why this defaults to sequential
        # (1) rather than to a specific higher number; a node operator that knows the
        # endpoint can take more can raise it through the environment
        max_concurrency = get_env_var("SPARQL_MAX_CONCURRENCY", 1, as_type="int")
        if max_concurrency < 1:
            safe_log(
                "warn",
                f"SPARQL_MAX_CONCURRENCY of {max_concurrency} is not valid; using 1 "
                f"(sequential) instead.",
            )
            max_concurrency = 1

        tasks = [
            (
                variable,
                partial(
                    _process_variable_query,
                    endpoint,
                    query_template,
                    variable,
                    variable_property,
                    schema,
                    use_schema,
                ),
            )
            for variable in variables_to_extract
        ]

        failed_variables = []
        for variable, result_df, error in _execute_concurrently(tasks, max_concurrency):
            if error is not None:
                failed_variables.append(variable)
                safe_log(
                    "error",
                    f"Query for {variable} failed and will be skipped, the rest of "
                    f"the extraction continues without it: {error}",
                )
                continue

            # A result is only ever None alongside an error, which is handled (and
            # skipped) above, so a result reaching this point is never None
            assert result_df is not None

            # Merging every result into the accumulated DataFrame as soon as it
            # arrives, rather than collecting every result first and merging them
            # afterwards, keeps the peak memory usage bound to the accumulated
            # result and the single largest query result, regardless of how many
            # variables are extracted
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

        if failed_variables and len(failed_variables) == len(variables_to_extract):
            raise AlgorithmError(
                f"Query failed for every requested variable: "
                f"{', '.join(failed_variables)}."
            )

    elif query_type == "multi_column":
        query_template = _prepare_query_template(
            _load_query_template("multi_column"), schema
        )

        try:
            intermediate_df = _process_multi_column_query(
                endpoint,
                query_template,
                variables_to_extract,
                variable_property,
                schema,
                use_schema,
            )
        except Exception as e:
            raise AlgorithmError(f"Error processing multi-column query: {e}")

    else:
        raise UserInputError(f"Unknown query type: {query_type}.")

    # Replace the missing value notation to prevent TypeErrors
    intermediate_df = intermediate_df.replace(missing_data_notation, pd.NA)

    # Count the missing values once every notation of missing data is represented as
    # such. Counting the dataset's own notation alone would leave the values that no
    # record holds at all - the ones that the merge of the variables leaves empty -
    # uncounted, which would make an absence of data an observation that is not
    # actually observed.
    add_missing_data_info(intermediate_df, pd.NA)

    # Sort by patient_id to ensure consistent ordering; identifiers are text, so they
    # are ordered naturally rather than lexicographically
    if not intermediate_df.empty and "patient_id" in intermediate_df.columns:
        intermediate_df = intermediate_df.sort_values(
            "patient_id", key=lambda identifiers: identifiers.map(_natural_sort_key)
        ).reset_index(drop=True)

    return intermediate_df
