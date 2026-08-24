"""
------------------------------------------------------------------------------
RDF/SPARQL Querying Functions

File organisation:
- SPARQL client (post_sparql_query)
------------------------------------------------------------------------------
"""

import csv
import json
import time
import requests  # type: ignore

from io import StringIO
from typing import Any, Dict, List, Union, Optional

from vantage6.algorithm.tools.exceptions import AlgorithmError
from vantage6.algorithm.tools.util import get_env_var
from vantage6_strongaya_general.miscellaneous import safe_log

# The number of seconds that a request may take before it is considered to have
# failed; deliberately kept a bit on the generous side, as an RDF-store can take a
# while to answer a query over a large graph
DEFAULT_TIMEOUT = 60

# The number of times that a failed request is retried before it is given up on
DEFAULT_MAX_RETRIES = 3

# HTTP status codes that reflect a transient failure of the endpoint rather than a
# problem with the request itself, and are therefore worth retrying
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


def post_sparql_query(
    endpoint: str,
    query: str,
    request_type: str = "query",
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    log_label: Optional[str] = None,
) -> Union[str, List[Dict[str, Any]], Dict[Any, Any]]:
    """
    Send a POST request to the specified endpoint with the given query.

    A request that cannot reach the endpoint at all (a connection error or a timeout)
    or that receives a server error (a 5xx status code) is retried, with an
    exponentially increasing delay between attempts; a request is retried a limited
    number of times only, so that a persistently failing endpoint eventually surfaces
    as an error rather than being retried forever. A status code that reflects a
    problem with the request itself (a 4xx status code) is not retried, as retrying it
    would not change the outcome.

    Args:
        endpoint (str): The URL of the endpoint to send the request to.
        query (str): The SPARQL query to send in the request body.
        request_type (str, optional): The type of request to send. Defaults to "query".
        headers (dict, optional): Any additional headers to include in the request.
        timeout (float, optional): The number of seconds that the request may take
                                   before it is considered to have failed. A value
                                   specified in the environment variables will be
                                   prioritised. Defaults to 60 seconds.
        max_retries (int, optional): The maximum number of times that a failed request
                                     is retried. A value specified in the environment
                                     variables will be prioritised. Defaults to 3.
        log_label (str, optional): A label that identifies the request within the log,
                                   used to make retries and failures traceable back to
                                   the query that caused them. Defaults to a generic
                                   label when not provided.

    Returns:
        Union[str, List[Dict[str, Any]], Dict[Any, Any]]: The server's response to the request.
    """
    if headers is None:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {request_type: query}

    if timeout is None:
        timeout = get_env_var("SPARQL_TIMEOUT", DEFAULT_TIMEOUT, as_type="int")
    if max_retries is None:
        max_retries = get_env_var(
            "SPARQL_MAX_RETRIES", DEFAULT_MAX_RETRIES, as_type="int"
        )

    label = log_label or "SPARQL request"
    response = None

    for attempt in range(max_retries + 1):
        is_last_attempt = attempt == max_retries
        try:
            response = requests.post(
                endpoint, data=data, headers=headers, timeout=timeout
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if is_last_attempt:
                raise AlgorithmError(
                    f"{label} could not reach {endpoint} after {max_retries + 1} "
                    f"attempt(s): {e}."
                )
            _wait_before_retry(label, endpoint, attempt, max_retries, str(e))
            continue

        if response.status_code == 200:
            break

        if response.status_code in RETRYABLE_STATUS_CODES and not is_last_attempt:
            _wait_before_retry(
                label,
                endpoint,
                attempt,
                max_retries,
                f"status code {response.status_code}",
            )
            continue

        raise AlgorithmError(
            f"{label} failed with status code {response.status_code} after "
            f"{attempt + 1} attempt(s)."
        )

    # A successful response always breaks out of the loop above, so this point is
    # only reached once one has been received
    assert response is not None

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        try:
            file_like_object = StringIO(response.text)
            reader = csv.DictReader(file_like_object)
            return list(reader)
        except Exception as e:
            raise AlgorithmError(
                f"{label} did not return a valid JSON or CSV, error: {e}.",
            )


def _wait_before_retry(
    label: str, endpoint: str, attempt: int, max_retries: int, reason: str
) -> None:
    """
    Report a retryable failure and wait before the next attempt is made.

    The delay doubles with every attempt (1s, 2s, 4s, ...), which spreads the retries
    of a widely used endpoint out over time instead of hammering it with attempts in
    quick succession.

    Args:
        label (str): A label that identifies the request within the log.
        endpoint (str): The endpoint that the request was sent to.
        attempt (int): The (zero-based) attempt that failed.
        max_retries (int): The maximum number of retries that are attempted.
        reason (str): A description of why the attempt failed.
    """
    backoff_seconds = 2**attempt
    safe_log(
        "warn",
        f"{label} failed against {endpoint} ({reason}); attempt {attempt + 1} of "
        f"{max_retries + 1}, retrying in {backoff_seconds}s.",
    )
    time.sleep(backoff_seconds)
