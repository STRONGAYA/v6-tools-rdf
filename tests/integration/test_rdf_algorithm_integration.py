"""
Comprehensive Vantage6 integration testing.
"""

import pytest
import json
import pandas as pd

from json import JSONDecodeError
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
from vantage6.algorithm.tools.exceptions import (
    DataError,
    UserInputError,
    CollectResultsError,
    PrivacyThresholdViolation,
    InputError,
    AlgorithmError,
    CollectOrganizationError,
)


@pytest.fixture
def test_methods():
    """
    Fixture providing different algorithm methods to test with their specific kwargs templates.

    GUIDANCE FOR REUSE:
    - Add your algorithm's method names as keys to this dict
    - Each method should specify its kwargs templates for different test scenarios
    - Method-specific parameters should be defined here, not in configurations
    - Ensure your algorithm supports all methods and parameters listed here

    STRUCTURE:
    Returns a dict where:
    - Keys are method names that can be called via vantage6 task input
    - Values are dicts containing kwargs templates for different test scenarios

    KWARGS TEMPLATES:
    Each method has kwargs for different test scenarios:
    - 'basic': Basic functionality test
    - 'organisation_selection': Test with specific organisations
    - 'data_stratification': Test with data stratification
    - 'inlier_specific': Test with inlier-specific variable configurations
    - 'return_partial': Test with partial results (method-specific)
    - 'parameter_galore': Test with all parameters combined

    DYNAMIC PARAMETER FILLING:
    Parameters set to None are automatically filled by test methods from configurations:
    - "variables_to_extract": Filled from config['variables_to_extract']
    - "query_type": Filled from config['query_type']

    EXAMPLES:
    - For statistical algorithms: {"central": {...}, "partial_general_statistics": {...}}
    - For ML algorithms: {"train": {...}, "predict": {...}, "validate": {...}}
    - For data processing: {"preprocess": {...}, "transform": {...}, "aggregate": {...}}
    """
    return {
        "partial_rdf_mock": {
            "basic": {
                "variables_to_extract": None,  # Will be filled from config
                "query_type": None,  # Will be filled from config if needed
            },
        },
    }


@pytest.fixture
def test_configurations(rdf_store):
    """
    Fixture providing comprehensive test configurations for algorithm validation.

    GUIDANCE FOR REUSE:
    1. MODIFY CONFIGURATIONS: Update each configuration to match your algorithm's requirements
    2. DATABASE LABELS: Change 'database_label' to match your test databases
    3. VARIABLES: Update variable specifications to match your data schema
    4. FAILURE SCENARIOS: Add configurations that test error handling and edge cases
    5. METHOD-SPECIFIC KWARGS: Now handled in test_methods fixture - keep configurations clean

    CONFIGURATION STRUCTURE:
    Each configuration dict should contain:
    - 'database_label': String identifying the test database
    - 'variables_to_extract': Dict of variables to extract, with their datatype
    - 'query_type': The type of query to use ('single_column' or 'multi_column')
    - 'expected_failure': Boolean indicating if this config should fail
    - 'failure_reason': String describing why failure is expected
    - 'expected_error_type': Exception class or list of exception classes expected on failure

    DYNAMIC PARAMETER FILLING:
    These values are used to fill None parameters in method kwargs:
    - 'variables_to_extract' -> "variables_to_extract"
    - 'query_type' -> "query_type"

    EXAMPLE CONFIGURATION TYPES:
    - 'standard_dataset': Normal successful execution
    - '*_bad_actor': Stress testing with resource constraints
    - '*_incorrect_input': Input validation testing
    - 'rare_dataset': Edge case with minimal data
    - 'non_existent_*': Error handling validation
    """
    return {
        "standard_dataset": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "ncit:C28421": {
                    "datatype": "categorical",
                },
                "ncit:C156420": {
                    "datatype": "numerical",
                },
            },
            "query_type": "single_column",
        },
        "standard_dataset_multi_column": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "ncit:C28421": {
                    "datatype": "categorical",
                },
                "ncit:C156420": {
                    "datatype": "numerical",
                },
            },
            "query_type": "multi_column",
        },
        "standard_dataset_linked_tables_multi_column": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                # Biological sex is held by the first table, whereas the time of PROM
                # recording is held by the second table
                "ncit:C28421": {
                    "datatype": "categorical",
                },
                "ncit:C192402": {
                    "datatype": "numerical",
                },
            },
            "query_type": "multi_column",
        },
        "standard_dataset_intermediate_class_variable": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                # The time of PROM recording is nested within the PROM container class
                "ncit:C192402": {
                    "datatype": "numerical",
                },
            },
            "query_type": "single_column",
        },
        "standard_dataset_bad_actor": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "<http://example.org/predicate> UNION { SERVICE <http://malicious.endpoint/sparql> "
                "{ SELECT ?data WHERE { ?s ?p ?data } } }": {"datatype": "categorical"},
            },
            "query_type": "single_column",
            "expected_failure": True,
            "failure_reason": "Invalid query injection.",
            "expected_error_type": [UserInputError, AlgorithmError],
        },
        "standard_dataset_incorrect_input": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "Variable_1": {"datatype": "categorical"},
                "ncit:C-does-not-exist": {"datatype": "numerical"},
            },  # Use a non-existent variable
            "query_type": "single_column",
            "expected_failure": True,
            "failure_reason": "Non-existent variables requested or invalid input structure specified",
            "expected_error_type": [DataError, AlgorithmError],
        },
        "standard_dataset_missing_variable_input": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "ncit:C0123456789": {"datatype": "categorical"},
            },  # Use a non-existent variable
            "query_type": "single_column",
        },
        "non_existent_dataset_standard_input": {
            "database_label": "not_my_rdf_store",  # Use a non-existent database
            "variables_to_extract": {
                "ncit:C28421": {
                    "datatype": "categorical",
                },
                "ncit:C156420": {
                    "datatype": "numerical",
                },
            },
            "query_type": "single_column",
            "expected_failure": True,
            "failure_reason": "Attempting to query an unknown database",
            "expected_error_type": [JSONDecodeError],
        },
    }


@pytest.mark.integration
class TestAlgorithmComponent:
    """
    Comprehensive test class for algorithm functionality across different methods and configurations.

    IMPORTANT NOTE FOR REUSE:
    This test class is specifically designed for descriptive statistics algorithms and must be
    adapted when repurposing for other algorithm types. The test functions contain algorithm-
    specific logic for input preparation, result extraction, and validation.

    WHEN REPURPOSING THIS CODE:
    1. REVIEW ALL TEST METHODS: Each test method contains algorithm-specific logic
    2. UPDATE KWARGS PREPARATION: Modify how kwargs are prepared from configurations
    3. ADAPT RESULT EXTRACTION: Update extract_data_from_result() for your algorithm's output
    4. MODIFY ASSERTIONS: Change validation logic to match your algorithm's expected behaviour
    5. UPDATE ERROR HANDLING: Ensure exception types match your algorithm's error patterns
    6. CONFIGURE DATABASE LABELS: Ensure test databases match your algorithm's requirements

    TEST SCENARIOS COVERED:
    - Basic functionality testing across all methods
    - Organisation-specific testing (federated learning scenarios)
    - Data stratification testing (subset analysis)
    - Error handling and edge case validation
    - Resource constraint testing (memory, computation limits)

    PARAMETRISATION:
    Tests are parametrised by:
    - method: Algorithm method to test (from test_methods fixture)
    - config_name: Configuration scenario (from test_configurations fixture)

    This creates a test matrix covering all combinations of methods × configurations.

    CUSTOMISATION CHECKLIST:
    □ Update kwargs preparation logic for your algorithm's parameters
    □ Modify database labels to match your test environment
    □ Adapt variable specifications to your data schema
    □ Update result extraction logic in extract_data_from_result()
    □ Modify assertion logic in determine_statistics_acceptance()
    □ Add algorithm-specific error types and handling
    □ Update test descriptions and naming conventions
    """

    @pytest.mark.parametrize("method", ["partial_rdf_mock"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_multi_column",
            "standard_dataset_linked_tables_multi_column",
            "standard_dataset_intermediate_class_variable",
            "standard_dataset_incorrect_input",
            "standard_dataset_bad_actor",
            "standard_dataset_missing_variable_input",
            "non_existent_dataset_standard_input",
        ],
    )
    def test_algorithm_basic(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with different methods and configurations, including expected failures.

        CUSTOMISATION REQUIRED:
        - Update kwargs preparation for your algorithm's parameter structure
        - Modify task creation parameters as needed
        - Adapt result validation logic
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]
        # Prepare method-specific kwargs from method configuration
        # This dynamically fills kwargs based on the test configuration and method
        kwargs = method_config["basic"].copy()
        kwargs["variables_to_extract"] = config["variables_to_extract"]
        kwargs["query_type"] = config["query_type"]

        # Create a task for the client to retrieve the descriptive data
        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} algorithm run - {config_name}",
            image=algorithm_image_name,
            description=f"Task to test the {method} function using {config_name} configuration.",
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            # Test that aggressive configurations fail gracefully
            with pytest.raises(Exception) as exc_info:
                extracted_values = extract_data_from_result(client, task)

            # Verify specific error types (support both single error type and list of error types)
            expected_errors = config.get("expected_error_type")
            if expected_errors:
                # Convert single error type to list for uniform handling
                if not isinstance(expected_errors, list):
                    expected_errors = [expected_errors]

                # Check if the raised exception matches any of the expected types
                error_matched = any(
                    isinstance(exc_info.value, expected_error)
                    for expected_error in expected_errors
                )
                assert error_matched, (
                    f"Expected one of {[err.__name__ for err in expected_errors]} "
                    f"but got {type(exc_info.value).__name__}"
                )

            print(f"Expected failure occurred for {config_name}: {exc_info.value}")
        else:
            # Normal success path
            extracted_values = extract_data_from_result(client, task)
            assert determine_result_acceptance(
                extracted_values, kwargs
            ), f"Extracted values did not match expected values for {config_name} configuration"


def extract_data_from_result(client, task) -> List[pd.DataFrame]:
    """
    Extract and validate data from the algorithm task result.

    :param client: Authenticated Vantage6 client
    :param task: Task object containing task details
    :return: List[pd.DataFrame] list with extracted variable values as dataframes
    """
    # Wait for results to be ready
    print("Waiting for results")
    task_id = task["id"]
    result = client.wait_for_results(task_id)

    # Check if there are any (un-)expected errors in the log
    run_info = client.run.from_task(task_id)
    log = run_info["data"][0]["log"]

    if "Traceback" in log:
        print(f"Error found in task log: {log}")

        # Extract the actual error from the log
        error_lines = [line for line in log.split("\n") if line.startswith("error >")]
        if error_lines:
            # Look for traceback information
            if "Traceback" in log:
                # Extract the exception type and message from the traceback
                lines = log.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("vantage6.algorithm.tools.exceptions."):
                        error_class_line = line.strip()
                        error_message = (
                            error_class_line.split(": ", 1)[1]
                            if ": " in error_class_line
                            else "Unknown error"
                        )
                        if "UserInputError" in error_class_line:
                            raise UserInputError(error_message)
                        elif "CollectResultsError" in error_class_line:
                            raise CollectResultsError(error_message)
                        elif "PrivacyThresholdViolation" in error_class_line:
                            raise PrivacyThresholdViolation(error_message)
                        elif "InputError" in error_class_line:
                            raise InputError(error_message)
                        elif "AlgorithmError" in error_class_line:
                            raise AlgorithmError(error_message)
                        elif "CollectOrganizationError" in error_class_line:
                            raise CollectOrganizationError(error_message)
                        elif "DataError" in error_class_line:
                            raise DataError(error_message)
                        else:
                            # If the error class is not recognised, raise a generic AlgorithmError
                            raise AlgorithmError(
                                f"Unknown error type in log: {error_class_line}"
                            )

        # Fallback to generic error with the error message
        error_message = error_lines[-1].replace("error >", "").strip()
        if error_message and error_message != "None":
            raise AlgorithmError(f"Algorithm execution failed: {error_message}")

    # Check if the result is not None
    assert result is not None, "Result should not be None"

    # Extract the aggregated results
    result = json.loads(result["data"][0]["result"])

    # Handle list of JSON strings (from one or multiple organizations)
    dataframes = []
    if isinstance(result, list):
        for json_string in result:
            # Parse each JSON string into a DataFrame
            df = pd.read_json(StringIO(json_string))
            dataframes.append(df)
    else:
        # Single JSON string result
        df = pd.read_json(StringIO(result))
        dataframes.append(df)

    # Check if results are present
    assert dataframes is not None, "No results could be retrieved from the result."

    return dataframes


def normalise_values(values: pd.Series) -> pd.Series:
    """
    Represent values as comparable strings, with a single notation for missing values.

    :param values: pd.Series holding the values to normalise
    :return: pd.Series holding the normalised values
    """
    return (
        values.astype(str)
        .replace({"None": "", "nan": "", "NaN": "", "<NA>": "", "none": ""})
        .str.strip()
    )


def determine_coverage_acceptance(
    federated_result: List[pd.DataFrame],
    coverage_variables: List[str],
    expected_df: pd.DataFrame,
) -> bool:
    """
    Validate results of variables that the expected data holds no values for.

    The expected data holds the values of the first table only, which is why the
    variables of the second table - such as the time of PROM recording, which is nested
    within the PROM container class - are validated on their coverage: every patient of
    the expected data should hold a value for them. The variables that the expected data
    does hold are still compared to their expected values.

    :param federated_result: List[pd.DataFrame] with the extracted values
    :param coverage_variables: The variables that the expected data holds no values for
    :param expected_df: pd.DataFrame with the expected data
    :return: bool indicating whether the results are accepted
    """
    for index, result_df in enumerate(federated_result):
        for variable in coverage_variables:
            if variable not in result_df.columns:
                print(f"Validation failed: DataFrame {index} lacks column '{variable}'")
                return False

            missing_values = int(result_df[variable].isna().sum())
            if missing_values:
                print(
                    f"Validation failed: DataFrame {index} holds {missing_values} "
                    f"missing value(s) for '{variable}'"
                )
                return False

        result_patients = set(normalise_values(result_df["patient_id"]))
        expected_patients = set(normalise_values(expected_df["patient_id"]))
        if result_patients != expected_patients:
            print(
                f"Validation failed: DataFrame {index} holds "
                f"{len(result_patients)} patient(s) instead of "
                f"{len(expected_patients)}"
            )
            return False

        # Values of the variables that the expected data does hold should still match
        merged = result_df.merge(
            expected_df, on="patient_id", suffixes=("_result", "_expected")
        )
        for variable in expected_df.columns:
            if variable == "patient_id" or variable not in result_df.columns:
                continue

            differences = int(
                (
                    normalise_values(merged[f"{variable}_result"])
                    != normalise_values(merged[f"{variable}_expected"])
                ).sum()
            )
            if differences:
                print(
                    f"Validation failed: DataFrame {index} holds {differences} "
                    f"differing value(s) for '{variable}'"
                )
                return False

    print(
        f"Validation passed: All {len(federated_result)} DataFrames hold the "
        f"expected coverage"
    )
    return True


def determine_result_acceptance(
    federated_result: List[pd.DataFrame], algorithm_kwargs: Dict[str, Any] | None = None
) -> bool:
    """
    Validate that federated RDF extraction results meet expected criteria.
    """
    if federated_result is None:
        print("Validation failed: Result is None")
        return False

    if algorithm_kwargs is None:
        algorithm_kwargs = {}

    query_type = algorithm_kwargs.get("query_type")
    if query_type in ["single_column", "multi_column"]:
        csv_path = Path(__file__).parent.parent / "data" / "data.csv"
        if csv_path.exists():
            # Load the expected data directly - it's already in the correct format
            expected_df = pd.read_csv(csv_path)
            
            # Convert string "None" to actual None/pd.NA to match our processing
            expected_df["ncit:C28421"] = expected_df["ncit:C28421"].apply(
                lambda x: None if x == "None" else x
            )
            
            # Convert string "NaN" to actual NaN for numerical column
            expected_df["ncit:C156420"] = pd.to_numeric(expected_df["ncit:C156420"], errors="coerce")
            
            expected_data = [expected_df]
        else:
            raise FileNotFoundError(f"Expected data file not found: {csv_path}")
    else:
        print(f"Validation skipped: Unsupported query_type '{query_type}'")
        return True

    try:
        if isinstance(federated_result, list):
            result_dataframes = federated_result
        else:
            result_dataframes = [federated_result]

        # Special case: non-existing variable
        if algorithm_kwargs.get("variables_to_extract") == {
            "ncit:C0123456789": {"datatype": "categorical"},
        }:
            for df in result_dataframes:
                assert (
                    df.empty
                ), "Result DataFrame should be empty for non-existing variable"
            return True

        # Special case: variables that the expected data holds no values for
        requested_variables = list(
            (algorithm_kwargs.get("variables_to_extract") or {}).keys()
        )
        unexpected_variables = [
            variable
            for variable in requested_variables
            if variable not in expected_data[0].columns
        ]
        if unexpected_variables:
            return determine_coverage_acceptance(
                result_dataframes, unexpected_variables, expected_data[0]
            )

        if len(result_dataframes) != len(expected_data):
            print(
                f"Validation failed: Expected {len(expected_data)} DataFrames, got {len(result_dataframes)}"
            )
            return False

        for i, (result_df, expected_df) in enumerate(
            zip(result_dataframes, expected_data)
        ):
            if not isinstance(result_df, pd.DataFrame):
                print(f"Validation failed: Result {i} is not a DataFrame")
                return False

            try:
                # Convert None to pd.NA in both dataframes for consistent comparison
                result_df_filled = result_df.copy()
                expected_df_filled = expected_df.copy()
                
                for col in result_df_filled.columns:
                    if col != "patient_id":
                        # Convert None to pd.NA in result
                        result_df_filled[col] = result_df_filled[col].apply(
                            lambda x: pd.NA if x is None else x
                        )
                        # Convert None to pd.NA in expected
                        expected_df_filled[col] = expected_df_filled[col].apply(
                            lambda x: pd.NA if x is None else x
                        )
                
                pd.testing.assert_frame_equal(
                    result_df_filled,
                    expected_df_filled,
                    check_dtype=False,
                    check_like=True,
                    rtol=1e-5,
                    atol=1e-8,
                )
                print(f"Validation passed: DataFrame {i} matches expected data")
            except AssertionError as e:
                print(f"Validation failed: DataFrame {i} does not match expected data")
                print(f"Difference details: {e}")
                
                # Debug: Show a sample of the differences
                print("\nDebug info:")
                print(f"Result shape: {result_df.shape}, Expected shape: {expected_df.shape}")
                print(f"Result columns: {list(result_df.columns)}")
                print(f"Expected columns: {list(expected_df.columns)}")
                
                # Show first few rows where they differ
                if result_df.shape == expected_df.shape:
                    diff_mask = result_df != expected_df
                    if diff_mask.any().any():
                        print("\nFirst few differing rows:")
                        diff_rows = result_df[diff_mask.any(axis=1)].head()
                        expected_diff_rows = expected_df[diff_mask.any(axis=1)].head()
                        print("Result:")
                        print(diff_rows)
                        print("Expected:")
                        print(expected_diff_rows)
                
                return False

        print(
            f"Validation passed: All {len(result_dataframes)} DataFrames match expected data"
        )
        return True

    except Exception as e:
        print(f"Validation failed with exception: {e}")
        return False
