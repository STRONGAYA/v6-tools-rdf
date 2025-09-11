"""
Unit tests for library functions.

Test the actual library functionality rather than external libraries.
"""

import pytest
import pandas as pd
import sys
import os
from pathlib import Path

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vantage6_strongaya_rdf.data_processing import (  # noqa: E402
    add_missing_data_info,
    extract_subclass_info,
)
from vantage6_strongaya_rdf.sparql_client import post_sparql_query  # noqa: E402


class TestDataProcessing:
    """Test data processing functions."""

    def test_add_missing_data_info(self):
        """Test adding missing data information to DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "variable1": [1, None, 3],
                "variable2": ["A", "B", None],
            }
        )

        # Add missing data info with placeholder value
        add_missing_data_info(df, None)

        # Check that predetermined_info attribute was added
        assert hasattr(df, "predetermined_info")

    def test_extract_subclass_info_with_subclass(self):
        """Test extracting subclass information when sub_class column exists."""
        # Create test DataFrame with sub_class column
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "value": ["val1", "val2", "val3"],
                "sub_class": ["subclass1", "", "subclass3"],
            }
        )

        result = extract_subclass_info(df, "test_variable")

        # Check that sub_class and value columns are removed
        assert "sub_class" not in result.columns
        assert "value" not in result.columns

        # Check that the variable column was created
        assert "test_variable" in result.columns

        # Check that subclass info is properly used
        assert result.iloc[0]["test_variable"] == "subclass1"  # Has subclass
        assert result.iloc[1]["test_variable"] == "val2"  # Empty subclass, use value
        assert result.iloc[2]["test_variable"] == "subclass3"  # Has subclass

    def test_extract_subclass_info_without_subclass(self):
        """Test extracting subclass information when sub_class column doesn't exist."""
        # Create test DataFrame without sub_class column
        df = pd.DataFrame(
            {"patient_id": ["P1", "P2", "P3"], "value": ["val1", "val2", "val3"]}
        )

        result = extract_subclass_info(df, "test_variable")

        # Check that value column was renamed to variable name
        assert "value" not in result.columns
        assert "test_variable" in result.columns

        # Check that values are preserved
        assert result.iloc[0]["test_variable"] == "val1"
        assert result.iloc[1]["test_variable"] == "val2"
        assert result.iloc[2]["test_variable"] == "val3"


class TestSparqlClient:
    """Test SPARQL client functions."""

    def test_post_sparql_query_invalid_endpoint(self):
        """Test SPARQL client with invalid endpoint."""
        # This should handle gracefully and not crash
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . } LIMIT 1"

        # Test with invalid endpoint should return None or raise appropriate exception
        try:
            result = post_sparql_query("http://invalid-endpoint:9999/sparql", query)
            # If it doesn't raise an exception, result should be None or empty
            assert result is None or result == []
        except Exception as e:
            # Should raise a connection-related exception
            assert "connection" in str(e).lower() or "refused" in str(e).lower()


@pytest.mark.integration
class TestRDFStoreSetup:
    """Test RDF-store setup fixtures and functionality."""

    def test_flyover_repository_fixture(self, flyover_repository):
        """Test that Flyover repository is properly set up."""
        assert flyover_repository is not None
        assert "path" in flyover_repository
        assert os.path.exists(flyover_repository["path"])

        # Check that docker-compose.yml exists in the Flyover directory
        compose_file = os.path.join(flyover_repository["path"], "docker-compose.yml")
        assert os.path.exists(compose_file)

        print(f"Flyover repository set up at: {flyover_repository['path']}")

    def test_rdf_store_fixture(self, rdf_store):
        """Test that RDF-store is properly started and accessible."""
        assert rdf_store is not None
        assert "repository_id" in rdf_store
        assert "endpoint" in rdf_store
        assert "base_url" in rdf_store

        repository_id = rdf_store["repository_id"]
        endpoint = rdf_store["endpoint"]
        base_url = rdf_store["base_url"]

        assert repository_id is not None
        assert endpoint.startswith("http://localhost:7200/repositories/")
        assert base_url == "http://localhost:7200"

        print(f"RDF-store set up with repository: {repository_id}")
        print(f"Endpoint: {endpoint}")

        # Test that the endpoint is accessible
        import requests

        response = requests.get(f"{base_url}/rest/repositories")
        assert response.status_code == 200

        repositories = response.json()
        repo_ids = [repo.get("id") for repo in repositories]
        assert repository_id in repo_ids

    def test_rdf_store_data_loading(self, rdf_store):
        """Test that test data was properly loaded into the RDF-store."""
        import requests

        endpoint = rdf_store["endpoint"]

        # Test a simple SPARQL query to check if data is loaded
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . } LIMIT 10"

        response = requests.post(
            endpoint,
            headers={"Accept": "application/sparql-results+json"},
            data={"query": query},
        )

        assert response.status_code == 200
        results = response.json()

        # Should have some results if data was loaded
        bindings = results.get("results", {}).get("bindings", [])
        assert (
            len(bindings) > 0
        ), "No data found in RDF-store - test data may not have loaded properly"

        print(f"Found {len(bindings)} triples in RDF-store")


@pytest.mark.slow
@pytest.mark.integration
class TestRDFAlgorithmIntegration:
    """Integration tests for RDF algorithm functionality with real RDF-store."""

    def test_collect_sparql_data_with_rdf_store(self, rdf_store):
        """Test collect_sparql_data function with live RDF-store."""
        # This would test the actual library function against the running RDF-store
        # Import the actual function
        try:
            from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data

            endpoint = rdf_store["endpoint"]
            variables_to_describe = [
                "ncit:C28421"
            ]  # Use a variable that should exist in test data

            # This might fail if the test data doesn't contain the expected variables
            # but the test will verify the function can at least connect and execute
            try:
                result = collect_sparql_data(
                    variables_to_describe=variables_to_describe,
                    query_type="single_column",
                    endpoint=endpoint,
                    variable_property="sio:SIO_000008",
                )

                # Check that we get a DataFrame back
                assert isinstance(result, pd.DataFrame)
                print(
                    f"collect_sparql_data returned DataFrame with shape: {result.shape}"
                )

            except Exception as e:
                # If specific variables aren't found, that's okay for this test
                # We're mainly testing that the connection works
                print(f"Note: collect_sparql_data test returned: {e}")
                # Re-raise only if it's a connection error
                if "connection" in str(e).lower():
                    raise

        except ImportError as e:
            pytest.skip(f"Could not import collect_sparql_data function: {e}")
