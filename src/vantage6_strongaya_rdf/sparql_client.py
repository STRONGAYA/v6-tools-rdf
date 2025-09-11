"""
------------------------------------------------------------------------------
RDF/SPARQL Querying Functions

File organisation:
- SPARQL client (post_sparql_query)
------------------------------------------------------------------------------
"""

import csv
import json
import requests  # type: ignore

from io import StringIO
from typing import Any, Dict, List, Union, Optional

from vantage6.algorithm.tools.exceptions import AlgorithmError


def post_sparql_query(
    endpoint: str,
    query: str,
    request_type: str = "query",
    headers: Optional[Dict[str, str]] = None,
) -> Union[str, List[Dict[str, Any]], Dict[Any, Any]]:
    """
    Send a POST request to the specified endpoint with the given query.

    Args:
        endpoint (str): The URL of the endpoint to send the request to.
        query (str): The SPARQL query to send in the request body.
        request_type (str, optional): The type of request to send. Defaults to "query".
        headers (dict, optional): Any additional headers to include in the request.

    Returns:
        Union[str, List[Dict[str, Any]], Dict[Any, Any]]: The server's response to the request.
    """
    if headers is None:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {request_type: query}
    response = requests.post(endpoint, data=data, headers=headers)
    if response.status_code != 200:
        raise AlgorithmError(
            f"SPARQL request failed with status code {response.status_code}."
        )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        try:
            file_like_object = StringIO(response.text)
            reader = csv.DictReader(file_like_object)
            return list(reader)
        except Exception as e:
            raise AlgorithmError(
                f"SPARQL request did not return a valid JSON or CSV, error: {e}.",
            )
