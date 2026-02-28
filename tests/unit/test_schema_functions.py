"""
Unit tests for schema loader and parser functions.
"""

import sys

from pathlib import Path

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vantage6_strongaya_rdf.schema_loader import load_schema  # noqa: E402
from vantage6_strongaya_rdf.schema_parser import (  # noqa: E402
    build_predicate_path,
    get_variable_query_params,
    resolve_intermediate_class_path,
)


class TestSchemaLoader:
    """Test schema loader functions."""

    def test_load_bundled_schema(self):
        """Test loading schema from bundled resource."""
        schema = load_schema(use_remote=False)

        # Check that schema was loaded
        assert schema is not None
        assert isinstance(schema, dict)

        # Check basic schema structure
        assert "schema" in schema
        assert "variables" in schema["schema"]

    def test_load_schema_with_invalid_remote_fallback(self):
        """Test that invalid remote URL falls back to bundled schema."""
        schema = load_schema(
            use_remote=True,
            schema_url="http://invalid-url-that-does-not-exist.com/schema.jsonld",
            local_fallback=True,
        )

        # Should still get the bundled schema
        assert schema is not None
        assert isinstance(schema, dict)


class TestSchemaParser:
    """Test schema parser functions."""

    @classmethod
    def setup_class(cls):
        """Load schema once for all tests."""
        cls.schema = load_schema(use_remote=False)

    def test_build_predicate_path_simple(self):
        """Test building predicate path for a simple variable."""
        # Test with a variable that has schema reconstruction
        path = build_predicate_path("gender", self.schema)

        # Should return a property path
        assert path is not None
        assert isinstance(path, str)
        assert path.startswith("(")
        assert path.endswith(")*")
        assert "|" in path  # Should contain multiple predicates
        assert "dbo:has_column" in path  # Should include dbo:has_column

    def test_build_predicate_path_with_intermediate_class(self):
        """Test building predicate path for variable with intermediate class."""
        # time_prom_recording has PROM as intermediate class
        path = build_predicate_path("time_prom_recording", self.schema)

        # Should return a property path
        assert path is not None
        assert isinstance(path, str)
        assert path.startswith("(")
        assert path.endswith(")*")

    def test_build_predicate_path_nonexistent_variable(self):
        """Test building predicate path for non-existent variable."""
        path = build_predicate_path("nonexistent_variable", self.schema)

        # Should return empty string
        assert path == ""

    def test_resolve_intermediate_class_path_prom(self):
        """Test resolving intermediate class path for PROM class."""
        # PROM class: ncit:C177377
        path = resolve_intermediate_class_path("ncit:C177377", self.schema)

        # Should return a list of predicates
        assert isinstance(path, list)
        # Should have at least one predicate
        assert len(path) > 0

    def test_resolve_intermediate_class_path_ehr(self):
        """Test resolving intermediate class path for EHR class."""
        # EHR class: ncit:C142529
        path = resolve_intermediate_class_path("ncit:C142529", self.schema)

        # Should return a list of predicates
        assert isinstance(path, list)
        # Should have at least one predicate
        assert len(path) > 0

    def test_resolve_intermediate_class_path_nonexistent(self):
        """Test resolving path for non-existent intermediate class."""
        path = resolve_intermediate_class_path("ncit:NONEXISTENT", self.schema)

        # Should return empty list
        assert isinstance(path, list)
        assert len(path) == 0

    def test_get_variable_query_params(self):
        """Test getting query parameters for a variable."""
        params = get_variable_query_params("gender", self.schema)

        # Check that all required parameters are present
        assert "predicate_path" in params
        assert "main_class" in params
        assert "ontology_prefix" in params

        # Check parameter types
        assert isinstance(params["predicate_path"], str)
        assert isinstance(params["main_class"], str)
        assert isinstance(params["ontology_prefix"], str)

        # Check that main_class has the correct format
        assert ":" in params["main_class"]

        # Check that ontology_prefix ends with ":"
        assert params["ontology_prefix"].endswith(":")

        # Check that predicate path is properly formatted
        assert params["predicate_path"].startswith("(")
        assert params["predicate_path"].endswith(")*")

    def test_get_variable_query_params_by_class_code(self):
        """Test looking up query parameters using class code instead of variable name."""
        params = get_variable_query_params("ncit:C28421", self.schema)

        # Should find biological_sex by class code
        assert params != {}
        assert params["main_class"] == "ncit:C28421"
        assert params["ontology_prefix"] == "ncit:"
        assert params["predicate_path"].startswith("(")
        assert params["predicate_path"].endswith(")*")

    def test_get_variable_query_params_by_class_code_numerical(self):
        """Test looking up query parameters using class code for numerical variable."""
        params = get_variable_query_params("ncit:C156420", self.schema)

        # Should find age_at_initial_diagnosis by class code
        assert params != {}
        assert params["main_class"] == "ncit:C156420"
        assert params["ontology_prefix"] == "ncit:"

    def test_predicate_path_includes_dbo_has_column(self):
        """Test that predicate paths include dbo:has_column for database ontology compatibility."""
        path = build_predicate_path("biological_sex", self.schema)

        assert "dbo:has_column" in path
        assert "sio:SIO_000008" in path

    def test_get_variable_query_params_nonexistent(self):
        """Test getting query parameters for non-existent variable."""
        params = get_variable_query_params("nonexistent_variable", self.schema)

        # Should return empty dict
        assert params == {}

    def test_get_variable_query_params_age_at_initial_diagnosis(self):
        """Test query parameters for age_at_initial_diagnosis variable."""
        params = get_variable_query_params("age_at_initial_diagnosis", self.schema)

        # Should have all parameters
        assert params["main_class"] == "ncit:C156420"
        assert params["ontology_prefix"] == "ncit:"

        # Predicate path should contain multiple predicates due to EHR intermediate class
        path = params["predicate_path"]
        assert path.startswith("(")
        assert path.endswith(")*")
        # Verify expected predicates are in the path with proper namespace prefix
        assert "dbo:has_column" in path, f"Expected 'dbo:has_column' in path: {path}"
        assert "sio:SIO_000255" in path, f"Expected 'sio:SIO_000255' in path: {path}"
        assert "sio:SIO_000008" in path, f"Expected 'sio:SIO_000008' in path: {path}"
        # Verify pipe separator between predicates
        assert "|" in path, f"Expected pipe separator in path: {path}"


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise just import to check for errors
    try:
        import pytest

        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic import check")
        print("All imports successful!")
