"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python manual_vantage_mock.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""

import sys

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path

# Get path of current directory
current_path = Path(__file__).parent

# Install the local v6-tools-rdf library in editable mode
repo_root = current_path.parent.parent
print("Checking whether v6-tools-rdf is importable...")
try:
    import vantage6_strongaya_rdf  # noqa: F401

    print("v6-tools-rdf is already installed.")
except ImportError:
    print(
        "v6-tools-rdf not found, please install it using 'pip install -e .' "
        "in the directory that contains the 'pyproject.toml' file."
    )

# Add the mock algorithm to Python path so it can be imported
mock_algorithm_path = current_path.parent / "mock_algorithm" / "v6-rdf-mock"
sys.path.insert(0, str(mock_algorithm_path))

# Mock client
client = MockAlgorithmClient(
    datasets=[
        # Data for the first organisation
        [
            {
                "database": current_path / "test_data_rdf.csv",
                "db_type": "csv",
                "input_data": {},
            }
        ],
        # Data for the second organisation
        [
            {
                "database": current_path / "test_data_rdf.csv",
                "db_type": "csv",
                "input_data": {},
            }
        ],
    ],
    module="v6-rdf-mock",
)

# List mock organisations
organisations = client.organization.list()
print(organisations)
org_ids = [organisation["id"] for organisation in organisations]

# Run the partial method for all organisations
task = client.task.create(
    input_={
        "method": "partial_rdf_mock",
        "kwargs": {
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
    },
    organizations=org_ids,
)
print(task)

# Get the results from the task
results = client.wait_for_results(task.get("id"))
print(results)
