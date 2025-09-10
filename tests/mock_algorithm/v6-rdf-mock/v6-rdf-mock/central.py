from typing import Any, List

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

from vantage6_strongaya_general.miscellaneous import collect_organisation_ids, safe_log
from vantage6_strongaya_rdf.miscellaneous import check_input_structure


# TODO add British english Docstring and update any existing info logging; keep it concise
@algorithm_client
def central_rdf_mock(client: AlgorithmClient, variables_to_extract: List[str]) -> Any:
    """Central part of the algorithm"""
    safe_log("info", "Starting central algorithm function")

    # Collect all organisations that participate in this collaboration unless specified
    organisation_ids = collect_organisation_ids(None, client)

    input_ = {
        "method": "partial_rdf_mock",
        "kwargs": {
            "variables_to_extract": variables_to_extract,
        },
    }

    # create a subtask for all organisations in the collaboration.
    safe_log("info", "Creating subtask for all organisations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organisation_ids,
        name="rdf-mock subtask",
        description="This is a very important subtask",
    )

    # Wait for the node to return results of the subtask.
    safe_log("info", "Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    safe_log("info", "Results obtained!")

    # No aggregation necessary for this mock algorithm
    return results
