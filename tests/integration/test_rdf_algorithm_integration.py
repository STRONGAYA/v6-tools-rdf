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
    - "variables_to_describe": Filled from config['variables_to_describe_basic'] or
    config['variables_to_describe_inlier_specific']
    - "organisation_ids": Filled from config['organisation_subset']
    - "variables_to_stratify": Filled from config['variables_to_stratify']

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
    - 'variables_to_describe_basic': Dict of variables for basic testing
    - 'organisation_subset': List of organisation IDs to test with
    - 'variables_to_stratify': Dict defining stratification parameters (optional)
    - 'expected_failure': Boolean indicating if this config should fail
    - 'failure_reason': String describing why failure is expected
    - 'expected_error_type': Exception class or list of exception classes expected on failure

    DYNAMIC PARAMETER FILLING:
    These values are used to fill None parameters in method kwargs:
    - 'variables_to_describe_basic' -> "variables_to_describe"
    - 'organisation_subset' -> "organisation_ids"
    - 'variables_to_stratify' -> "variables_to_stratify"

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
                }
            },
            "query_type": "single_column",
        },
        "standard_dataset_bad_actor": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "<http://example.org/predicate> UNION { SERVICE <http://malicious.endpoint/sparql> "
                "{ SELECT ?data WHERE { ?s ?p ?data } } }": {
                    "datatype": "categorical"
                },
            },
            "query_type": "single_column",
            "expected_failure": True,
            "failure_reason": "Invalid query injection.",
            "expected_error_type": [UserInputError, AlgorithmError],
        },
        "standard_dataset_incorrect_input": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "Variable_1": {
                    "datatype": "categorical"
                },
                "ncit:C-does-not-exist": {
                    "datatype": "numerical"
                },
            },  # Use a non-existent variable
            "query_type": "single_column",
            "expected_failure": True,
            "failure_reason": "Non-existent variables requested or invalid input structure specified",
            "expected_error_type": [DataError, AlgorithmError],
        },
        "standard_dataset_missing_variable_input": {
            "database_label": "rdf_store",  # Always use rdf_store as this refers to the RDF-store setup
            "variables_to_extract": {
                "ncit:C0123456789": {
                    "datatype": "categorical"
                },
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
                }
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


def determine_result_acceptance(
        federated_result: List[pd.DataFrame], algorithm_kwargs: Dict[str, Any] = None
) -> bool:
    """
    Validate that federated RDF extraction results meet expected criteria.

    This function performs validation of the federated computation results
    by checking data structure, content consistency, and comparing against
    expected mock data values.

    Args:
        federated_result: Results from federated computation (DataFrame or list of DataFrames)
        algorithm_kwargs: Dict of algorithm parameters used for the computation

    Returns:
        bool: True if results pass validation checks, False otherwise
    """

    if federated_result is None:
        print("Validation failed: Result is None")
        return False

    query_type = algorithm_kwargs.get("query_type")
    if query_type == "single_column":
        # Load expected data from CSV file
        csv_path = Path(__file__).parent.parent / "data" / "data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["patient_id"] = range(len(df))
            df = df.map(lambda x: x if not isinstance(x, (int, float)) else str(x))
            expected_data = [df.to_json(orient='columns')]
        else:
            raise FileNotFoundError(f"Expected data file not found: {csv_path}")

    else:
        print(f"Validation skipped: Unsupported query_type '{query_type}'")
        return True
    try:
        # Handle both single DataFrame and list of DataFrames
        if isinstance(federated_result, list):
            result_dataframes = federated_result
        else:
            result_dataframes = [federated_result]

        if algorithm_kwargs.get("variables_to_extract") == {
                "ncit:C0123456789": {
                    "datatype": "categorical"
                },
            }:
            for df in result_dataframes:
                assert (
                    df.empty
                ), "Result DataFrame should be empty for non-existing variable"
            return True

        # Convert expected data to DataFrames for comparison
        expected_dataframes = []
        for json_string in expected_data:
            expected_df = pd.read_json(StringIO(json_string))
            expected_dataframes.append(expected_df)

        # Validate we have the expected number of DataFrames
        if len(result_dataframes) != len(expected_dataframes):
            print(
                f"Validation failed: Expected {len(expected_dataframes)} DataFrames, got {len(result_dataframes)}"
            )
            return False

        # Validate each DataFrame
        for i, (result_df, expected_df) in enumerate(
                zip(result_dataframes, expected_dataframes)
        ):
            if not isinstance(result_df, pd.DataFrame):
                print(f"Validation failed: Result {i} is not a DataFrame")
                return False

            # Check shape
            if result_df.shape != expected_df.shape:
                print(
                    f"Validation failed: DataFrame {i} shape mismatch. "
                    f"Expected {expected_df.shape}, got {result_df.shape}"
                )
                return False

            # Check columns
            if not all(col in result_df.columns for col in expected_df.columns):
                print(f"Validation failed: DataFrame {i} missing expected columns")
                return False

            # Check a sample of data values for key columns
            if (
                    "patient_id" in result_df.columns
                    and "patient_id" in expected_df.columns
            ):
                if not result_df["patient_id"].equals(expected_df["patient_id"]):
                    print(
                        f"Validation failed: DataFrame {i} patient_id values don't match"
                    )
                    return False

            # For the variables, check that we have the expected unique values
            for variable in ["ncit:C28421", "ncit:C156420"]:
                if variable in result_df.columns and variable in expected_df.columns:
                    if not result_df[variable].equals(expected_df[variable]):
                        print(
                            f"Validation failed: DataFrame {i} values for {variable} don't match"
                        )
                        return False

        print(
            f"Validation passed: All {len(result_dataframes)} DataFrames match expected data"
        )
        return True

    except Exception as e:
        print(f"Validation failed with exception: {e}")
        return False
