from typing import Any, List

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

from vantage6_strongaya_general.miscellaneous import collect_organisation_ids, safe_log
from vantage6_strongaya_rdf.miscellaneous import check_input_structure


@algorithm_client
def central_rdf_mock(client: AlgorithmClient, variables_to_extract: List[str]) -> Any:
    """
    Central coordination function for the RDF mock algorithm.
    
    This function orchestrates the federated RDF data extraction process by 
    distributing subtasks to participating organisations and collecting their 
    results. It serves as a demonstration algorithm for testing RDF-based 
    federated learning infrastructure.
    
    Args:
        client: Vantage6 algorithm client for task coordination
        variables_to_extract: List of variable identifiers to extract from RDF stores
        
    Returns:
        Aggregated results from all participating organisations
    """
    safe_log("info", "Initiating central RDF mock algorithm")

    # Collect all organisations that participate in this collaboration unless specified
    organisation_ids = collect_organisation_ids(None, client)

    input_ = {
        "method": "partial_rdf_mock",
        "kwargs": {
            "variables_to_extract": variables_to_extract,
        },
    }

    # Create a subtask for all organisations in the collaboration.
    safe_log("info", "Creating subtask for participating organisations")
    task = client.task.create(
        input_=input_,
        organizations=organisation_ids,
        name="rdf-mock subtask",
        description="This is a very important subtask",
    )

    # Wait for the node to return results of the subtask.
    safe_log("info", "Awaiting results from organisations")
    results = client.wait_for_results(task_id=task.get("id"))
    safe_log("info", "Results successfully obtained")

    # No aggregation necessary for this mock algorithm
    return results
