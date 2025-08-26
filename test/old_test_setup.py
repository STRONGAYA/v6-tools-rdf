import unittest
import subprocess
import requests
import time
import pandas as pd
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Determine the project root directory
# If run from /test directory, go up one level
if os.path.basename(os.getcwd()) == "test":
    PROJECT_ROOT = os.path.dirname(os.getcwd())
else:
    PROJECT_ROOT = os.getcwd()

# Add the src directory to Python path for local imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# TODO move rdf-store setup using Flyover and data curl to conftest.py for reuse in other test files;
#  data is already in tests/data/*.ttl


class TestVantage6RDF(unittest.TestCase):
    """
    Test suite for v6-tools-rdf library using Flyover's GraphDB setup.

    This test:
    1. Clones the Flyover repository (if not already available)
    2. Uses docker-compose from Flyover to start GraphDB
    3. Runs tests against the GraphDB instance
    4. Cleans up when done
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment with GraphDB."""
        print(
            f"=== Starting test setup at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} ==="
        )
        print(f"Current directory: {os.getcwd()}")
        print(f"Project root: {PROJECT_ROOT}")

        # Verify imports from local files will work
        cls._verify_local_files()

        # Clone Flyover repository for docker-compose
        cls._clone_flyover_repo()

        # Start GraphDB using docker-compose from Flyover
        cls._start_graphdb_with_compose()

        # Wait for GraphDB to start
        cls._wait_for_graphdb()

        # Get repository
        cls.repository_id = cls._get_repository()

        # Load test data if needed
        cls._load_test_data()

        print(f"Setup complete. Using repository: {cls.repository_id}")

    @classmethod
    def tearDownClass(cls):
        """Clean up by stopping GraphDB and removing Flyover clone."""
        print(
            f"=== Cleaning up at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} ==="
        )

        # Stop GraphDB using docker-compose
        if hasattr(cls, "flyover_path") and cls.flyover_path:
            try:
                # Change to the Flyover directory
                original_dir = os.getcwd()
                os.chdir(cls.flyover_path)

                print("Stopping GraphDB service...")
                compose_cmd = cls._get_compose_command()
                if compose_cmd:
                    # Use the correct compose command format
                    if isinstance(compose_cmd, list):
                        cmd = compose_cmd + ["stop", "rdf-store"]
                    else:
                        cmd = [compose_cmd, "stop", "rdf-store"]

                    subprocess.run(
                        cmd,
                        check=False,  # Don't fail if already stopped
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    print("GraphDB service stopped")

                # Return to original directory
                os.chdir(original_dir)
            except Exception as e:
                print(f"Warning: Error during service cleanup: {e}")
                print("You may need to manually clean up containers.")

        # Remove temporary Flyover clone if we created one
        if hasattr(cls, "temp_dir") and cls.temp_dir and os.path.exists(cls.temp_dir):
            print(f"Removing temporary Flyover clone at: {cls.temp_dir}")
            try:
                shutil.rmtree(cls.temp_dir)
                print("Temporary directory removed successfully")
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory: {e}")

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

    @classmethod
    def _clone_flyover_repo(cls):
        """Clone the Flyover repository to use its docker-compose."""
        # Check if FLYOVER_PATH is set and valid
        flyover_path = os.environ.get("FLYOVER_PATH")
        if flyover_path and os.path.exists(flyover_path):
            if os.path.exists(os.path.join(flyover_path, "docker-compose.yml")):
                print(f"Using existing Flyover repository at: {flyover_path}")
                cls.flyover_path = flyover_path
                cls.temp_dir = None  # No temporary directory to clean up
                return

        # Create a temporary directory
        cls.temp_dir = tempfile.mkdtemp(prefix="flyover-temp-")
        cls.flyover_path = os.path.join(cls.temp_dir, "Flyover")

        print(f"Creating temporary directory: {cls.temp_dir}")

        # Clone repository
        print("Cloning Flyover repository...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/MaastrichtU-CDS/Flyover.git",
                    cls.flyover_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"Flyover repository cloned successfully to: {cls.flyover_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone Flyover repository: {e}")
            print(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else ''}")
            raise RuntimeError(
                "Failed to clone Flyover repository. Make sure Git is installed and working."
            )

    @classmethod
    def _get_compose_command(cls):
        """Determine the correct docker-compose command to use."""
        try:
            # Check for docker-compose command
            compose_cmd = "docker-compose"
            result = subprocess.run(
                [compose_cmd, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                print(f"Using docker-compose: {result.stdout.strip()}")
                return compose_cmd
        except FileNotFoundError:
            pass

        try:
            # Try docker compose command (newer Docker versions)
            compose_cmd = ["docker", "compose"]
            result = subprocess.run(
                compose_cmd + ["--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                print(f"Using docker compose: {result.stdout.strip()}")
                return compose_cmd
        except FileNotFoundError:
            pass

        print("Docker Compose not found. Please install Docker Compose.")
        raise RuntimeError("Docker Compose is required but not available")

    @classmethod
    def _start_graphdb_with_compose(cls):
        """Start GraphDB using docker-compose from Flyover."""
        if not cls.flyover_path:
            raise RuntimeError("Flyover path not set. Cannot start GraphDB.")

        # Get the appropriate compose command
        compose_cmd = cls._get_compose_command()
        if not compose_cmd:
            raise RuntimeError("Docker Compose command not available")

        try:
            # Change to the Flyover directory
            original_dir = os.getcwd()
            os.chdir(cls.flyover_path)

            # Check if docker-compose.yml exists
            if not os.path.exists("docker-compose.yml"):
                raise FileNotFoundError(
                    f"docker-compose.yml not found in {cls.flyover_path}"
                )

            # Check if service is already running
            if isinstance(compose_cmd, list):
                ps_cmd = compose_cmd + ["ps", "rdf-store"]
            else:
                ps_cmd = [compose_cmd, "ps", "rdf-store"]

            result = subprocess.run(
                ps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if "Up" in result.stdout:
                print("GraphDB service is already running")
            else:
                # Start only the GraphDB service
                print("Starting GraphDB service using docker-compose...")

                # Use the correct compose command format
                if isinstance(compose_cmd, list):
                    cmd = compose_cmd + ["up", "-d", "rdf-store"]
                else:
                    cmd = [compose_cmd, "up", "-d", "rdf-store"]

                process = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                if process.returncode != 0:
                    print(f"Docker Compose output: {process.stdout}")
                    print(f"Docker Compose error: {process.stderr}")
                    raise RuntimeError("Failed to start GraphDB with docker-compose")

                print("GraphDB service started using docker-compose")

            # Return to original directory
            os.chdir(original_dir)

        except Exception as e:
            # Make sure we return to the original directory even if there's an error
            if "original_dir" in locals():
                os.chdir(original_dir)
            raise e

    @classmethod
    def _get_test_data_path(cls, filename):
        """Get the path to a test data file, checking both test dir and project root."""
        # Check current directory first
        if os.path.exists(filename):
            return filename

        # Check test directory if we're not already in it
        test_dir_path = os.path.join(PROJECT_ROOT, "test", filename)
        if os.path.exists(test_dir_path):
            return test_dir_path

        # Check project root
        project_root_path = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(project_root_path):
            return project_root_path

        # File not found in any location
        return None

    @classmethod
    def _wait_for_graphdb(cls, timeout=180):  # Extended timeout for startup
        """Wait for GraphDB to be ready to accept connections."""
        print("Waiting for GraphDB to start...")
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"GraphDB didn't start within {timeout} seconds")

            try:
                response = requests.get("http://localhost:7200/rest/repositories")
                if response.status_code == 200:
                    print(
                        f"GraphDB is up and running! (took {time.time() - start_time:.1f} seconds)"
                    )
                    time.sleep(5)  # Additional safety margin
                    return
            except requests.exceptions.ConnectionError:
                pass

            print(
                f"Still waiting for GraphDB... ({int(time.time() - start_time)}s elapsed)"
            )
            time.sleep(5)

    @classmethod
    def _get_repository(cls):
        """Get the first available repository or create a new one."""
        response = requests.get("http://localhost:7200/rest/repositories")
        if not response.ok:
            raise RuntimeError(f"Failed to get list of repositories: {response.text}")

        repositories = response.json()
        if not repositories:
            print("No repositories found. Creating a test repository...")
            return cls._create_repository()

        # List all available repositories
        print("Available repositories:")
        for repo in repositories:
            repo_id = repo.get("id")
            repo_title = repo.get("title", "No title")
            print(f"  - {repo_id}: {repo_title}")

        # Use the first repository
        repo_id = repositories[0].get("id")
        print(f"Using repository: {repo_id}")
        return repo_id

    @classmethod
    def _create_repository(cls):
        """Create a new test repository."""
        config = {"id": "test-repo", "title": "Test Repository", "type": "free"}

        response = requests.post("http://localhost:7200/rest/repositories", json=config)

        if not response.ok:
            raise RuntimeError(f"Failed to create repository: {response.text}")

        print("Repository 'test-repo' created successfully")
        return "test-repo"

    @classmethod
    def _load_test_data(cls):
        """Load test data files into GraphDB."""
        data_filenames = ["ontology.ttl", "data.ttl", "annotation.ttl"]

        # Get repository endpoint
        repo_endpoint = f"http://localhost:7200/repositories/{cls.repository_id}/"

        # Find and load each file
        files_loaded = False
        for filename in data_filenames:
            file_path = cls._get_test_data_path(filename)

            if not file_path:
                print(f"Warning: {filename} not found in any location. Skipping...")
                continue

            print(f"Loading {file_path}...")

            content_type = {
                ".owl": "application/rdf+xml",
                ".ttl": "text/turtle",
                ".nt": "application/n-triples",
                ".nq": "application/n-quads",
                ".trig": "application/trig",
                ".jsonld": "application/ld+json",
            }.get(Path(file_path).suffix, "text/turtle")

            try:
                subprocess.run(
                    [
                        "curl",
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/x-turtle",
                        "--data-binary",
                        f"@{filename}",
                        f"{repo_endpoint}/rdf-graphs/service?graph=http://{filename[:filename.rfind('.')]}.local/",
                    ]
                )

                print(f"Successfully loaded {file_path}")
                files_loaded = True
            except e as e:
                print(f"Failed to load {file_path}: {e}")

    # ========== TEST CASES ==========

    # TODO Check whether we need to keep these testcases with renewed setup or whether we just discard them
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
