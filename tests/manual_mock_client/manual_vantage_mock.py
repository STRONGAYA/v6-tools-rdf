"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python manual_vantage_mock.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path

# Get path of current directory
current_path = Path(__file__).parent

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
    module="v6-descriptive-statistics",
)

# List mock organisations
organisations = client.organization.list()
print(organisations)
org_ids = [organisation["id"] for organisation in organisations]

# Run the central method on one node and get the results
central_task = client.task.create(
    input_={
        "method": "central_rdf_mock",
        "kwargs": {
            "variables_to_extract": ["Variable_1", "Variable_2"],
        }
    },
    organizations=[org_ids[0]],
)
results = client.wait_for_results(central_task.get("id"))
print(results)

# Run the partial method for all organisations
task = client.task.create(
    input_={
        "method": "partial_rdf_mock",
        "kwargs": {"kwargs": {
            "variables_to_extract": ["Variable_1", "Variable_2"],
        }
        },
    },
    organizations=org_ids,
)
print(task)

# Get the results from the task
results = client.wait_for_results(task.get("id"))
print(results)
