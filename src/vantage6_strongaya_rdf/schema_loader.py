"""
------------------------------------------------------------------------------
Schema Loader Module

Provides functionality to load the AYA cancer JSON-LD schema from bundled
resources or remote URL with fallback support.
------------------------------------------------------------------------------
"""

import json
import requests
from importlib import resources
from typing import Optional
from vantage6_strongaya_general.miscellaneous import safe_log

# Default schema URL
SCHEMA_URL = (
    "https://raw.githubusercontent.com/STRONGAYA/AYA-cancer-semantic-map/"
    "refs/heads/main/AYA_cancer_schema.jsonld"
)


def load_schema(
    use_remote: bool = False,
    schema_url: Optional[str] = None,
    local_fallback: bool = True,
) -> dict:
    """
    Load the AYA cancer schema.

    Args:
        use_remote: If True, fetch from GitHub URL
        schema_url: Custom URL to fetch schema from (overrides default)
        local_fallback: If remote fetch fails, fall back to bundled schema

    Returns:
        Parsed JSON-LD schema as dictionary

    Raises:
        Exception: If schema cannot be loaded from any source
    """
    schema_data = None

    # Try remote fetch if requested
    if use_remote:
        url = schema_url or SCHEMA_URL
        safe_log("info", f"Attempting to fetch schema from remote URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            schema_data = response.json()
            safe_log("info", "Successfully loaded schema from remote URL")
            return schema_data
        except Exception as e:
            safe_log("warning", f"Failed to fetch schema from remote URL: {e}")
            if not local_fallback:
                raise Exception(
                    f"Failed to fetch remote schema and local fallback is disabled: {e}"
                )

    # Load from bundled resource
    if schema_data is None:
        safe_log("info", "Loading schema from bundled resource")
        try:
            with (
                resources.files("vantage6_strongaya_rdf")
                .joinpath("schemas")
                .joinpath("AYA_cancer_schema.jsonld")
                .open("r") as file
            ):
                schema_data = json.load(file)
            safe_log("info", "Successfully loaded schema from bundled resource")
            return schema_data
        except Exception as e:
            safe_log("error", f"Failed to load schema from bundled resource: {e}")
            raise Exception(f"Failed to load schema from any source: {e}")

    return schema_data
