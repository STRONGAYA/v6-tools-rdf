"""
Unit tests for schema loader and parser functions.
"""

import sys

from pathlib import Path

# Add src directory to path for importing library functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest  # noqa: E402

from vantage6_strongaya_rdf.schema_loader import (  # noqa: E402
    BUNDLED_SCHEMA_TAG,
    SCHEMA_URL,
    build_schema_url,
    get_schema_version,
    load_schema,
    validate_schema,
)
from vantage6_strongaya_rdf.schema_parser import (  # noqa: E402
    build_predicate_path,
    get_intermediate_classes,
    get_schema_prefixes,
    get_variable_query_params,
    resolve_intermediate_class_path,
)

# Predicates that the AYA cancer schema uses to describe its structure
SOCIODEMOGRAPHIC_PREDICATE = "sio:SIO_000255"
ATTRIBUTE_PREDICATE = "sio:SIO_000008"
CONTAINER_PREDICATE = "sio:SIO_000253"
MEASUREMENT_PREDICATE = "sio:SIO_000233"
IDENTIFIER_PREDICATE = "sio:SIO_000673"

# Intermediate (container) classes of the AYA cancer schema
PROM_CLASS = "ncit:C177377"
HCPROM_CLASS = "ncit:C142453"
EHR_CLASS = "ncit:C142529"


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

    def test_load_schema_without_fallback_raises(self):
        """Test that an unreachable remote schema raises when no fallback is allowed."""
        with pytest.raises(Exception):
            load_schema(
                use_remote=True,
                schema_url="http://invalid-url-that-does-not-exist.com/schema.jsonld",
                local_fallback=False,
            )

    def test_bundled_schema_originates_from_a_release(self):
        """Test that the bundled schema is the release that the library records."""
        schema = load_schema(use_remote=False)

        # The recorded tag is the version of the bundled schema, prefixed with a "v"
        assert BUNDLED_SCHEMA_TAG == f"v{get_schema_version(schema)}"

    def test_schema_url_refers_to_a_release_tag(self):
        """Test that remote schemas are retrieved from a release rather than a branch."""
        # A branch is a moving target, whereas a release is a versioned contract
        assert "refs/heads" not in SCHEMA_URL
        assert f"refs/tags/{BUNDLED_SCHEMA_TAG}" in SCHEMA_URL
        assert SCHEMA_URL.endswith("AYA_cancer_schema.jsonld")

    def test_build_schema_url(self):
        """Test composing the URL of a specific schema release."""
        url = build_schema_url("v9.9.9")

        assert "refs/tags/v9.9.9" in url
        assert url.endswith("AYA_cancer_schema.jsonld")

    def test_validate_bundled_schema(self):
        """Test that the bundled schema holds the expected structure."""
        # Should not raise
        validate_schema(load_schema(use_remote=False))

    def test_validate_schema_rejects_empty_document(self):
        """Test that a document without variables is rejected."""
        with pytest.raises(ValueError):
            validate_schema({})

    def test_validate_schema_rejects_unsupported_format(self):
        """Test that an earlier, non JSON-LD, revision of the semantic map is rejected.

        Such a document would otherwise be accepted silently, after which every
        variable would fall back to the default variable property.
        """
        legacy_schema = {
            "endpoint": "http://localhost:7200/repositories/userRepo/statements",
            "prefixes": "PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>",
            "variable_info": {
                "identifier": {
                    "data_type": "identifier",
                    "predicate": "sio:SIO_000673",
                    "class": "ncit:C25364",
                }
            },
        }

        with pytest.raises(ValueError):
            validate_schema(legacy_schema)

    def test_validate_schema_rejects_variables_without_predicates(self):
        """Test that variables that describe no structure are rejected."""
        with pytest.raises(ValueError):
            validate_schema({"schema": {"variables": {"identifier": {"label": "id"}}}})

    def test_get_schema_version_of_unversioned_schema(self):
        """Test that an unversioned schema does not break version reporting."""
        assert get_schema_version({}) == "unknown"


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

        # Should return the exact ordered sequence that the schema describes
        assert path == f"{SOCIODEMOGRAPHIC_PREDICATE}/{ATTRIBUTE_PREDICATE}"
        # dbo:has_column is only the fallback, not mixed into schema-derived paths
        assert "dbo:has_column" not in path

    def test_build_predicate_path_is_an_ordered_sequence(self):
        """Test that predicate paths are ordered sequences rather than transitive alternations.

        A transitive alternation such as "(a|b)*" matches the predicates in any order, at any
        depth and - because of the zero-length match - the patient node itself, which allows
        values of unrelated branches to be picked up.
        """
        for variable_name in [
            "gender",
            "biological_sex",
            "age_at_initial_diagnosis",
            "identifier",
            "time_prom_recording",
        ]:
            path = build_predicate_path(variable_name, self.schema)

            assert path, f"No path was built for '{variable_name}'"
            assert "*" not in path, f"Path of '{variable_name}' is transitive: {path}"
            assert "?" not in path, f"Path of '{variable_name}' is optional: {path}"
            assert "+" not in path, f"Path of '{variable_name}' is repeated: {path}"

    def test_build_predicate_path_identifier_is_a_single_predicate(self):
        """Test that the identifier's path is the bare predicate."""
        path = build_predicate_path("identifier", self.schema)

        assert path == IDENTIFIER_PREDICATE

    def test_build_predicate_path_with_intermediate_class(self):
        """Test building predicate path for variable with intermediate class.

        The PROM, EHR and HCPROM classes are attached to the node of another variable,
        which means that the path towards them consists of the path to that variable's
        node plus the hop into the container itself. Omitting that hop yields a path
        that cannot match any data.
        """
        # time_prom_recording is measured within the PROM container
        path = build_predicate_path("time_prom_recording", self.schema)

        # The route to the container is followed by the hop into it and the variable itself
        assert path.endswith(
            f"/{CONTAINER_PREDICATE}/{MEASUREMENT_PREDICATE}"
        ), f"Path does not reach the PROM container: {path}"

        # The route to the container's parent node is part of the path
        assert path.startswith(
            (SOCIODEMOGRAPHIC_PREDICATE, f"({SOCIODEMOGRAPHIC_PREDICATE}")
        ), f"Path does not start at the patient's characteristics: {path}"

        # The shortest route that the schema describes is amongst the alternatives
        shortest_route = f"{SOCIODEMOGRAPHIC_PREDICATE}/{ATTRIBUTE_PREDICATE}"
        assert (
            f"{shortest_route}/{CONTAINER_PREDICATE}" in path
            or f"{shortest_route}|" in path
        ), f"Shortest route to the PROM container is missing: {path}"

    def test_build_predicate_path_for_all_container_variables(self):
        """Test that every variable that is nested in a container reaches that container."""
        for variable_name in [
            "time_prom_recording",
            "time_hcprom_recording",
            "time_ehrprom_recording",
            "administered_prom_language",
        ]:
            path = build_predicate_path(variable_name, self.schema)

            assert (
                CONTAINER_PREDICATE in path
            ), f"Path of '{variable_name}' does not enter its container: {path}"
            assert (
                path.count("/") >= 3
            ), f"Path of '{variable_name}' is too short to reach its container: {path}"

    def test_get_intermediate_classes(self):
        """Test that the intermediate (container) classes are recognised."""
        intermediate_classes = get_intermediate_classes(self.schema)

        assert intermediate_classes == {PROM_CLASS, HCPROM_CLASS, EHR_CLASS}

    def test_get_intermediate_classes_handles_empty_schema(self):
        """Test that an empty schema holds no intermediate classes."""
        assert get_intermediate_classes({}) == set()

    def test_build_predicate_path_nonexistent_variable(self):
        """Test building predicate path for non-existent variable."""
        path = build_predicate_path("nonexistent_variable", self.schema)

        # Should return empty string
        assert path == ""

    def test_resolve_intermediate_class_path_prom(self):
        """Test resolving intermediate class path for PROM class."""
        # PROM class: ncit:C177377
        path = resolve_intermediate_class_path(PROM_CLASS, self.schema)

        # Should return the route towards the container and the hop into it
        assert isinstance(path, list)
        assert len(path) == 2
        assert path[-1] == CONTAINER_PREDICATE
        assert f"{SOCIODEMOGRAPHIC_PREDICATE}/{ATTRIBUTE_PREDICATE}" in path[0]

    def test_resolve_intermediate_class_path_ehr(self):
        """Test resolving intermediate class path for EHR class."""
        # EHR class: ncit:C142529
        path = resolve_intermediate_class_path(EHR_CLASS, self.schema)

        # Should return the route towards the container and the hop into it
        assert isinstance(path, list)
        assert len(path) == 2
        assert path[-1] == CONTAINER_PREDICATE
        assert f"{SOCIODEMOGRAPHIC_PREDICATE}/{ATTRIBUTE_PREDICATE}" in path[0]

    def test_resolve_intermediate_class_path_hcprom(self):
        """Test resolving intermediate class path for HCPROM class."""
        # HCPROM class: ncit:C142453
        path = resolve_intermediate_class_path(HCPROM_CLASS, self.schema)

        # Should return the route towards the container and the hop into it
        assert isinstance(path, list)
        assert len(path) == 2
        assert path[-1] == CONTAINER_PREDICATE

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
        assert params["predicate_path"] == (
            f"{SOCIODEMOGRAPHIC_PREDICATE}/{ATTRIBUTE_PREDICATE}"
        )

    def test_get_variable_query_params_by_class_code(self):
        """Test looking up query parameters using class code instead of variable name."""
        params = get_variable_query_params("ncit:C28421", self.schema)

        # Should find biological_sex by class code
        assert params != {}
        assert params["main_class"] == "ncit:C28421"
        assert params["ontology_prefix"] == "ncit:"
        assert params["predicate_path"] == (
            f"{SOCIODEMOGRAPHIC_PREDICATE}/{ATTRIBUTE_PREDICATE}"
        )

    def test_get_variable_query_params_by_class_code_numerical(self):
        """Test looking up query parameters using class code for numerical variable."""
        params = get_variable_query_params("ncit:C156420", self.schema)

        # Should find age_at_initial_diagnosis by class code
        assert params != {}
        assert params["main_class"] == "ncit:C156420"
        assert params["ontology_prefix"] == "ncit:"

    def test_predicate_path_does_not_include_dbo_has_column(self):
        """Test that schema-derived predicate paths do NOT include dbo:has_column.
        dbo:has_column is only used as a fallback when no schema predicates are found.
        """
        path = build_predicate_path("biological_sex", self.schema)

        assert "dbo:has_column" not in path
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

        # Predicate path should describe the route towards the variable's own node
        path = params["predicate_path"]
        # Verify expected predicates are in the path with proper namespace prefix
        assert "sio:SIO_000255" in path, f"Expected 'sio:SIO_000255' in path: {path}"
        assert "sio:SIO_000008" in path, f"Expected 'sio:SIO_000008' in path: {path}"
        # dbo:has_column is only the fallback, not mixed into schema-derived paths
        assert (
            "dbo:has_column" not in path
        ), f"dbo:has_column should not be in schema path: {path}"
        # Verify the sequence separator between predicates
        assert "/" in path, f"Expected sequence separator in path: {path}"

    def test_get_schema_prefixes_returns_dict(self):
        """Test that get_schema_prefixes returns a dictionary."""
        prefixes = get_schema_prefixes(self.schema)

        assert isinstance(prefixes, dict)
        assert len(prefixes) > 0

    def test_get_schema_prefixes_contains_expected_prefixes(self):
        """Test that get_schema_prefixes returns expected common prefixes."""
        prefixes = get_schema_prefixes(self.schema)

        # Check for prefixes that should be in the schema
        expected_prefixes = ["ncit", "sio", "mesh", "roo"]
        for prefix in expected_prefixes:
            assert (
                prefix in prefixes
            ), f"Expected prefix '{prefix}' not found in {list(prefixes.keys())}"

    def test_get_schema_prefixes_values_are_strings(self):
        """Test that all prefix values are strings (URIs)."""
        prefixes = get_schema_prefixes(self.schema)

        for key, value in prefixes.items():
            assert isinstance(
                value, str
            ), f"Prefix '{key}' value is not a string: {type(value)}"
            assert value.startswith(
                "http"
            ), f"Prefix '{key}' value does not look like a URI: {value}"

    def test_get_schema_prefixes_handles_empty_schema(self):
        """Test get_schema_prefixes with empty schema."""
        prefixes = get_schema_prefixes({})

        assert isinstance(prefixes, dict)
        assert len(prefixes) == 0

    def test_get_schema_prefixes_merges_context_and_schema_prefixes(self):
        """Test that prefixes from both @context and schema.prefixes are merged."""
        prefixes = get_schema_prefixes(self.schema)

        # Both @context and schema.prefixes should be included
        # @context has: sio, ncit, mesh, roo, xsd, schema, mapping
        # schema.prefixes has: mesh, sio, ncit, roo, strongaya, sct, gsso
        # Result should include all unique prefixes
        assert "strongaya" in prefixes
        assert "sct" in prefixes
        assert "gsso" in prefixes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
