"""
------------------------------------------------------------------------------
Schema Loader Module

Provides functionality to load the AYA cancer JSON-LD schema from bundled
resources or a tagged release with fallback support.
------------------------------------------------------------------------------
"""

import json
import requests
from importlib import resources
from typing import Optional
from vantage6_strongaya_general.miscellaneous import safe_log

# Upstream repository that publishes the AYA cancer semantic map
SCHEMA_REPOSITORY = "STRONGAYA/AYA-cancer-semantic-map"

# File name of the schema within that repository
SCHEMA_FILE_NAME = "AYA_cancer_schema.jsonld"

# Release tag of the schema that is bundled with this library
BUNDLED_SCHEMA_TAG = "v2.0.1"

# Locations from which a released schema can be retrieved
SCHEMA_TAG_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/{repository}/refs/tags/{tag}/{file_name}"
)
LATEST_RELEASE_URL = f"https://api.github.com/repos/{SCHEMA_REPOSITORY}/releases/latest"

# Default schema URL; the release that is bundled with this library
SCHEMA_URL = SCHEMA_TAG_URL_TEMPLATE.format(
    repository=SCHEMA_REPOSITORY, tag=BUNDLED_SCHEMA_TAG, file_name=SCHEMA_FILE_NAME
)

# Timeout in seconds for any request to the upstream repository
REQUEST_TIMEOUT = 10


def get_schema_version(schema: dict) -> str:
    """
    Retrieve the version of a schema.

    Args:
        schema: The full schema dictionary

    Returns:
        The schema's version, or "unknown" when the schema does not declare one
    """
    return schema.get("version") or schema.get("schema", {}).get("version") or "unknown"


def validate_schema(schema: dict) -> None:
    """
    Verify that a document holds the structure that the schema parser expects.

    Without this verification, a document of a different shape - for instance an
    older, non JSON-LD, revision of the semantic map or a GitHub error page - would
    be accepted silently, after which every variable would fall back to the
    default variable property and no reconstruction would take place.

    Args:
        schema: The document to verify

    Raises:
        ValueError: If the document does not hold the expected structure
    """
    if not isinstance(schema, dict):
        raise ValueError(f"Schema is not a JSON object but a {type(schema).__name__}")

    variables = schema.get("schema", {}).get("variables")

    if not isinstance(variables, dict) or not variables:
        raise ValueError("Schema does not contain any variables in 'schema.variables'")

    described_variables = [
        name
        for name, definition in variables.items()
        if isinstance(definition, dict)
        and definition.get("predicate")
        and definition.get("class")
    ]

    if not described_variables:
        raise ValueError(
            "Schema variables do not declare a 'predicate' and 'class'; "
            "the document is likely of an unsupported format"
        )

    safe_log(
        "info",
        f"Validated schema version {get_schema_version(schema)} "
        f"with {len(variables)} variables",
    )


def resolve_latest_schema_tag() -> Optional[str]:
    """
    Determine the tag of the most recent release of the semantic map.

    Returns:
        The release's tag (e.g. "v2.0.1"), or None when it could not be determined
    """
    try:
        response = requests.get(LATEST_RELEASE_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        tag = response.json().get("tag_name")
    except Exception as e:
        safe_log("warning", f"Failed to determine the latest schema release: {e}")
        return None

    if not tag:
        safe_log("warning", "Latest schema release does not declare a tag")
        return None

    safe_log("info", f"Latest schema release is tagged '{tag}'")
    return tag


def build_schema_url(schema_tag: str) -> str:
    """
    Compose the URL of the schema as published in a specific release.

    Args:
        schema_tag: The release's tag (e.g. "v2.0.1")

    Returns:
        URL of the schema within that release
    """
    return SCHEMA_TAG_URL_TEMPLATE.format(
        repository=SCHEMA_REPOSITORY, tag=schema_tag, file_name=SCHEMA_FILE_NAME
    )


def load_schema(
    use_remote: bool = False,
    schema_url: Optional[str] = None,
    schema_tag: Optional[str] = None,
    local_fallback: bool = True,
) -> dict:
    """
    Load the AYA cancer schema.

    Remote schemas are retrieved from a release of the semantic map rather than from
    its main branch, so that a node always runs against a versioned schema. When no
    tag is given, the most recent release is resolved; the release that is bundled
    with this library is used when that resolution is unsuccessful.

    Args:
        use_remote: If True, fetch the schema from the upstream repository
        schema_url: Custom URL to fetch schema from (overrides the release URL)
        schema_tag: Release tag to fetch the schema from (e.g. "v2.0.1");
                    defaults to the most recent release
        local_fallback: If remote fetch fails, fall back to bundled schema

    Returns:
        Parsed JSON-LD schema as dictionary

    Raises:
        Exception: If schema cannot be loaded from any source
    """
    schema_data = None

    # Try remote fetch if requested
    if use_remote:
        url = schema_url or build_schema_url(
            schema_tag or resolve_latest_schema_tag() or BUNDLED_SCHEMA_TAG
        )
        safe_log("info", f"Attempting to fetch schema from remote URL: {url}")
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            schema_data = response.json()
            validate_schema(schema_data)
            safe_log("info", "Successfully loaded schema from remote URL")
            return schema_data
        except Exception as e:
            safe_log("warning", f"Failed to fetch schema from remote URL: {e}")
            schema_data = None
            if not local_fallback:
                raise Exception(
                    f"Failed to fetch remote schema and local fallback is disabled: {e}"
                )

    # Load from bundled resource
    if schema_data is None:
        safe_log("info", f"Loading schema from bundled resource ({BUNDLED_SCHEMA_TAG})")
        try:
            with (
                resources.files("vantage6_strongaya_rdf")
                .joinpath("schemas")
                .joinpath(SCHEMA_FILE_NAME)
                .open("r") as file
            ):
                schema_data = json.load(file)
            validate_schema(schema_data)
            safe_log("info", "Successfully loaded schema from bundled resource")
            return schema_data
        except Exception as e:
            safe_log("error", f"Failed to load schema from bundled resource: {e}")
            raise Exception(f"Failed to load schema from any source: {e}")

    return schema_data
