import unittest
import requests
import pandas as pd
import os
import sys
from datetime import datetime

# Determine the project root directory
# If run from /test directory, go up one level
if os.path.basename(os.getcwd()) == "test":
    PROJECT_ROOT = os.path.dirname(os.getcwd())
else:
    PROJECT_ROOT = os.getcwd()

# Add the src directory to Python path for local imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# NOTE: This file is now deprecated in favour of the pytest-based test infrastructure in /tests
# The Flyover setup has been successfully migrated to conftest.py and new tests have been implemented
# covering the same functionality with better organisation and reusability.
# This file can be safely removed after ensuring all functionality is covered by the new tests.

class TestVantage6RDF(unittest.TestCase):
    """
    Test suite for v6-tools-rdf library using Flyover's GraphDB setup.

    This test class now expects GraphDB to be set up externally (via pytest fixtures).
    The Flyover setup logic has been moved to conftest.py for better reusability.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment - expects GraphDB to already be running."""
        print(
            f"=== Starting test setup at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} ==="
        )
        print(f"Current directory: {os.getcwd()}")
        print(f"Project root: {PROJECT_ROOT}")

        # Verify imports from local files will work
        cls._verify_local_files()

        # Check if GraphDB is available
        try:
            response = requests.get("http://localhost:7200/rest/repositories")
            if not response.ok:
                raise RuntimeError("GraphDB is not accessible. Make sure Flyover setup has been run.")

            repositories = response.json()
            if not repositories:
                raise RuntimeError("No repositories found in GraphDB.")

            cls.repository_id = repositories[0].get("id")
            print(f"Using repository: {cls.repository_id}")

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to GraphDB at localhost:7200. "
                "Make sure Flyover setup has been run and GraphDB is running."
            )

    @classmethod
    def tearDownClass(cls):
        """Clean up - no longer handles Flyover cleanup as it's managed by fixtures."""
        print(f"=== Test cleanup at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print("Note: Flyover cleanup is now handled by pytest fixtures")

    @classmethod
    def _verify_local_files(cls):
        """Verify that local source files are available."""
        required_paths = [
            os.path.join(
                PROJECT_ROOT, "src/vantage6_strongaya_rdf/collect_sparql_data.py"
            ),
            os.path.join(PROJECT_ROOT, "src/vantage6_strongaya_rdf/sparql_client.py"),
        ]

        # Query template path (verify it exists)
        template_file = os.path.join(
            PROJECT_ROOT,
            "src",
            "vantage6_strongaya_rdf",
            "query_templates",
            "single_column.rq",
        )
        required_paths.append(template_file)

        print("Verifying local source files...")
        missing_files = []

        for path in required_paths:
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            print(f"Warning: The following files are missing: {missing_files}")
            print(f"Project root directory contents: {os.listdir(PROJECT_ROOT)}")
            if os.path.exists(os.path.join(PROJECT_ROOT, "src")):
                print(
                    f"src directory contents: {os.listdir(os.path.join(PROJECT_ROOT, 'src'))}"
                )

                # Print subdirectories to help diagnose where template might be
                vantage6_dir = os.path.join(
                    PROJECT_ROOT, "src", "vantage6_strongaya_rdf"
                )
                if os.path.exists(vantage6_dir):
                    print(
                        f"vantage6_strongaya_rdf contents: {os.listdir(vantage6_dir)}"
                    )

                    template_dir = os.path.join(vantage6_dir, "query_templates")
                    if os.path.exists(template_dir):
                        print(f"query_templates contents: {os.listdir(template_dir)}")
                    else:
                        print("query_templates directory not found")

            # Fail if critical files are missing
            if any(x in path for x in ["collect_sparql_data.py", "sparql_client.py"]):
                raise FileNotFoundError(
                    f"Critical source files are missing: {missing_files}"
                )

            print("Non-critical files missing, continuing anyway.")

    # ========== DEPRECATED TEST CASES ==========
    # These test cases have been superseded by the new pytest-based infrastructure in /tests
    # The functionality is now covered by:
    # - tests/unit/test_library_functions.py (unit tests for SPARQL client and data processing)
    # - tests/integration/test_rdf_algorithm_integration.py (end-to-end algorithm tests)
    # - tests/conftest.py (RDF-store setup fixtures)
    #
    # These tests are kept temporarily for reference but should be removed once migration is confirmed
    def test_collect_sparql_data_output(self):
        """
        Test that collect_sparql_data returns consistent output.

        This test verifies that:
        1. The function returns a DataFrame
        2. The DataFrame has the expected columns
        3. The column values match expected values from the repository
        """
        # Import directly from local file
        from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data

        # Define variables to query based on your data
        variables_to_describe = ["ncit:C28421"]

        print(f"\nTesting collect_sparql_data with variables: {variables_to_describe}")

        # Execute the function being tested
        result = collect_sparql_data(
            variables_to_describe=variables_to_describe,
            query_type="single_column",
            endpoint=f"http://localhost:7200/repositories/{self.repository_id}",
            variable_property="sio:SIO_000008",
        )

        # Output result information
        print(f"Result DataFrame shape: {result.shape}")
        print(f"Result DataFrame columns: {result.columns.tolist()}")
        if not result.empty:
            print(f"First few rows:\n{result.head(3)}")

        # Verify DataFrame properties
        self.assertIsInstance(
            result, pd.DataFrame, "Result should be a pandas DataFrame"
        )
        self.assertFalse(result.empty, "Result DataFrame should not be empty")

        # Verify columns match expected variables
        all_cols = set(result.columns)
        self.assertIn("patient_id", all_cols, "patient_id column should be present")

        # Check that all requested variables are in the result
        for variable in variables_to_describe:
            self.assertIn(
                variable, all_cols, f"Column {variable} should be present in result"
            )

        # ===== Direct comparison with SPARQL results =====
        # Get raw data from GraphDB to compare with collect_sparql_data results

        # We'll execute a SPARQL query to get the same data directly
        print("\nVerifying results match direct SPARQL query...")

        # Query for each variable
        all_direct_results = {}

        for variable in variables_to_describe:
            # Extract ontology prefix and class name
            ontology_prefix = variable.split(":")[0]

            # Query to get the same data collect_sparql_data would fetch
            sparql_query = f"""
            PREFIX {ontology_prefix}: <http://dbpedia.org/ontology/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?patient ?value
            WHERE {{
                ?patient dbo:has_column ?obj .
                ?obj a {variable} .
                ?obj rdf:value ?value .
            }}
            """

            response = requests.post(
                f"http://localhost:7200/repositories/{self.repository_id}",
                headers={"Accept": "application/sparql-results+json"},
                data={"query": sparql_query},
            )

            if not response.ok:
                print(f"Failed to execute SPARQL query for {variable}: {response.text}")
                continue

            query_results = response.json()

            # Convert to dictionary: patient_uri -> value
            variable_data = {}
            for binding in query_results["results"]["bindings"]:
                patient_uri = binding["patient"]["value"]
                patient_id = patient_uri.split("/")[-1]  # Extract ID from URI
                value = binding["value"]["value"]
                variable_data[patient_id] = value

            all_direct_results[variable] = variable_data

            print(f"Direct query for {variable} found {len(variable_data)} patients")

        # Compare collect_sparql_data results with direct query results
        for variable, variable_data in all_direct_results.items():
            for patient_id, expected_value in variable_data.items():
                # Find this patient in result DataFrame
                patient_rows = result[result["patient_id"] == patient_id]

                if not patient_rows.empty:
                    actual_value = str(patient_rows.iloc[0][variable])

                    # Handle NA/None values
                    if pd.isna(actual_value):
                        actual_value = None

                    self.assertEqual(
                        actual_value,
                        expected_value,
                        f"Value mismatch for patient {patient_id}, variable {variable}: "
                        f"expected {expected_value}, got {actual_value}",
                    )
                else:
                    self.fail(
                        f"Patient {patient_id} from direct query not found in result DataFrame"
                    )

    def test_sparql_client(self):
        """Test the SPARQL client functionality."""
        # Import from local file
        from vantage6_strongaya_rdf.sparql_client import post_sparql_query

        print("\nTesting post_sparql_query function")

        # Simple SPARQL query
        query = """
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
        } LIMIT 10
        """

        # Execute the query
        result = post_sparql_query(
            endpoint=f"http://localhost:7200/repositories/{self.repository_id}",
            query=query,
        )

        print(f"SPARQL query result type: {type(result)}")
        print(f"Result contains data: {bool(result)}")

        # Validate response
        self.assertTrue(result, "SPARQL query should return non-empty result")
        self.assertTrue(
            isinstance(result, (list, dict)),
            f"Result should be list or dict, got {type(result)}",
        )

    def test_query_template_loading(self):
        """Test the query template loading functionality."""
        # Import from local file
        from vantage6_strongaya_rdf.collect_sparql_data import _load_query_template

        print("\nTesting query template loading")

        # Load the template used by collect_sparql_data
        template = _load_query_template("single_column")

        print(f"Template loaded successfully: {bool(template)}")
        if template:
            template_lines = template.split("\n")
            print(f"Template preview: {len(template_lines)} lines")
            print("\n".join(template_lines[:5]) + "\n[...]\n")

        # Validate
        self.assertTrue(template, "Query template should be loaded successfully")
        self.assertIn(
            "PLACEHOLDER",
            template,
            "Template should contain placeholder for variable substitution",
        )

    def test_variable_processing(self):
        """Test variable query processing with complete validation."""
        # Import from local files
        from vantage6_strongaya_rdf.collect_sparql_data import (
            _process_variable_query,
            _load_query_template,
        )

        print("\nTesting variable query processing")

        # Load query template
        query_template = _load_query_template("single_column")
        if not query_template:
            self.skipTest("Skipping test as query template could not be loaded")

        # Process a single variable
        variable = "ncit:C28421"  # Adjust based on your data
        endpoint = f"http://localhost:7200/repositories/{self.repository_id}"
        variable_property = "sio:SIO_000008"

        result_df = _process_variable_query(
            endpoint, query_template, variable, variable_property
        )

        print(f"Result for {variable}:")
        print(f"  Shape: {result_df.shape}")
        print(f"  Columns: {result_df.columns.tolist()}")

        # Verify DataFrame structure
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertFalse(result_df.empty, f"Should find data for {variable}")
        self.assertIn("patient_id", result_df.columns)
        self.assertIn(variable, result_df.columns)

        # Verify against direct SPARQL query to ensure consistency
        sparql_query = f"""
        PREFIX dbo: <http://um-cds/ontologies/databaseontology/>
        PREFIX roo: <http://www.cancerdata.org/roo/>
        PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX sio: <http://semanticscience.org/resource/>
        PREFIX sct: <http://snomed.info/id/>
        PREFIX strongaya: <http://strongaya.eu/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?patient ?sub_class (SAMPLE(?value) AS ?any_value)
        WHERE {{
            ?patient sio:SIO_000008 ?sub_class_type .
            ?sub_class_type rdf:type ?main_class .
            ?sub_class_type rdf:type ncit:C28421 .
            ?sub_class_type dbo:has_cell ?sub_cell .
            ?sub_cell dbo:has_value ?value .
            FILTER strStarts(str(?main_class), str(ncit:))
            BIND(strafter(str(?main_class), str(ncit:)) AS ?main_class_code)
            OPTIONAL {{
                ?sub_cell rdf:type ?sub_class .
                FILTER (strStarts(str(?sub_class), str(ncit:))||strStarts(str(?sub_class), str(ncit:))) .
                ?sub_class rdfs:subClassOf ?main_class .
                FILTER (!regex(str(?main_class), str(?sub_class))) .
            }}
        }}
        GROUP BY ?patient ?sub_class
        """

        response = requests.post(
            endpoint,
            headers={"Accept": "application/sparql-results+json"},
            data={"query": sparql_query},
        )

        if not response.ok:
            self.skipTest(
                f"Skipping detailed verification as direct query failed: {response.text}"
            )
            return

        query_results = response.json()
        direct_count = len(query_results["results"]["bindings"])

        print(f"Direct query found {direct_count} results")
        self.assertEqual(
            len(result_df),
            direct_count,
            f"Result should have same number of rows as direct query",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
