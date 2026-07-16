"""
Unit tests for library functions.

Test the actual library functionality rather than external libraries.
"""

import sys

import pandas as pd

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
                "any_value": ["val1", "val2", "val3"],
                "sub_class": ["subclass1", "", "subclass3"],
            }
        )

        result = extract_subclass_info(df, "test_variable")

        # Check that sub_class and value columns are removed
        assert "sub_class" not in result.columns
        assert "any_value" not in result.columns

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
            {"patient_id": ["P1", "P2", "P3"], "any_value": ["val1", "val2", "val3"]}
        )

        result = extract_subclass_info(df, "test_variable")

        # Check that value column was renamed to variable name
        assert "any_value" not in result.columns
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
