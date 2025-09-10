"""
Pytest configuration and fixtures for v6-descriptive-statistics tests.
"""

import os
import pytest
import docker
import time
import json
import platform
import subprocess
import tempfile
import shutil
import requests
from datetime import datetime

from pathlib import Path

# Optional import for vantage6 - only needed for vantage6 integration tests
try:
    from vantage6.client import UserClient as Client
    VANTAGE6_AVAILABLE = True
except ImportError:
    Client = None
    VANTAGE6_AVAILABLE = False


@pytest.fixture(scope="session")
def docker_test_setup():
    """Check if Docker is available for testing."""
    try:
        client = docker.from_env()
        client.ping()
        return {"docker_available": True, "client": client}
    except docker.errors.DockerException:
        return {"docker_available": False, "client": None}


@pytest.fixture(scope="session")
def docker_client():
    """Get the Docker client and determine Docker host for the entire test session."""
    try:
        client = docker.from_env()
        client.ping()

        # Determine the appropriate Docker host
        docker_host = get_docker_host()
        print(f"Detected Docker host: {docker_host}")

        # Add the docker_host as an attribute to the client for easy access
        client.docker_host = docker_host

        return client
    except docker.errors.DockerException:
        pytest.skip("Docker not available")


# ========== RDF-store setup fixtures using Flyover ==========

@pytest.fixture(scope="session")
def flyover_repository():
    """Clone or use existing Flyover repository for GraphDB setup."""
    # Check if FLYOVER_PATH is set and valid
    flyover_path = os.environ.get("FLYOVER_PATH")
    if flyover_path and os.path.exists(flyover_path):
        if os.path.exists(os.path.join(flyover_path, "docker-compose.yml")):
            print(f"Using existing Flyover repository at: {flyover_path}")
            yield {"path": flyover_path, "temp_dir": None}
            return

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="flyover-temp-")
    flyover_path = os.path.join(temp_dir, "Flyover")

    print(f"Creating temporary directory: {temp_dir}")
    print("Cloning Flyover repository...")

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/MaastrichtU-CDS/Flyover.git",
                flyover_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Flyover repository cloned successfully to: {flyover_path}")

        yield {"path": flyover_path, "temp_dir": temp_dir}

    except subprocess.CalledProcessError as e:
        print(f"Failed to clone Flyover repository: {e}")
        pytest.skip("Failed to clone Flyover repository. Make sure Git is available.")

    finally:
        # Clean up temporary directory if we created one
        if temp_dir and os.path.exists(temp_dir):
            print(f"Removing temporary Flyover clone at: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print("Temporary directory removed successfully")
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory: {e}")


def _get_compose_command():
    """Determine the correct docker-compose command to use."""
    try:
        # Check for docker-compose command
        compose_cmd = "docker-compose"
        result = subprocess.run(
            [compose_cmd, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print(f"Using docker-compose: {result.stdout.strip()}")
            return compose_cmd
    except FileNotFoundError:
        pass

    try:
        # Try docker compose command (newer Docker versions)
        compose_cmd = ["docker", "compose"]
        result = subprocess.run(
            compose_cmd + ["--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print(f"Using docker compose: {result.stdout.strip()}")
            return compose_cmd
    except FileNotFoundError:
        pass

    print("Docker Compose not found. Please install Docker Compose.")
    raise RuntimeError("Docker Compose is required but not available")


@pytest.fixture(scope="session")
def rdf_store(flyover_repository, docker_client):
    """Start GraphDB RDF-store using Flyover's docker-compose."""
    flyover_info = flyover_repository
    flyover_path = flyover_info["path"]

    if not flyover_path:
        pytest.skip("Flyover path not available. Cannot start RDF-store.")

    # Get the appropriate compose command
    try:
        compose_cmd = _get_compose_command()
    except RuntimeError as e:
        pytest.skip(str(e))

    # Start GraphDB service
    try:
        # Change to the Flyover directory
        original_dir = os.getcwd()
        os.chdir(flyover_path)

        # Check if docker-compose.yml exists
        if not os.path.exists("docker-compose.yml"):
            pytest.skip(f"docker-compose.yml not found in {flyover_path}")

        # Check if service is already running
        if isinstance(compose_cmd, list):
            ps_cmd = compose_cmd + ["ps", "rdf-store"]
        else:
            ps_cmd = [compose_cmd, "ps", "rdf-store"]

        result = subprocess.run(
            ps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if "Up" not in result.stdout:
            # Start only the GraphDB service
            print("Starting GraphDB service using docker-compose...")

            # Use the correct compose command format
            if isinstance(compose_cmd, list):
                cmd = compose_cmd + ["up", "-d", "rdf-store"]
            else:
                cmd = [compose_cmd, "up", "-d", "rdf-store"]

            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if process.returncode != 0:
                print(f"Docker Compose output: {process.stdout}")
                print(f"Docker Compose error: {process.stderr}")
                pytest.skip("Failed to start GraphDB with docker-compose")

            print("GraphDB service started using docker-compose")
        else:
            print("GraphDB service is already running")

        # Return to original directory
        os.chdir(original_dir)

        # Wait for GraphDB to be ready
        _wait_for_graphdb()

        # Get or create repository
        repository_id = _get_or_create_repository()

        # Load test data
        _load_test_data(repository_id)

        yield {
            "repository_id": repository_id,
            "endpoint": f"http://localhost:7200/repositories/{repository_id}",
            "base_url": "http://localhost:7200"
        }

    except Exception as e:
        # Make sure we return to the original directory even if there's an error
        if "original_dir" in locals():
            os.chdir(original_dir)
        pytest.skip(f"Failed to set up RDF-store: {e}")

    finally:
        # Clean up: stop GraphDB service
        try:
            original_dir = os.getcwd()
            os.chdir(flyover_path)

            print("Stopping GraphDB service...")
            # Use the correct compose command format
            if isinstance(compose_cmd, list):
                cmd = compose_cmd + ["stop", "rdf-store"]
            else:
                cmd = [compose_cmd, "stop", "rdf-store"]

            subprocess.run(
                cmd,
                check=False,  # Don't fail if already stopped
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("GraphDB service stopped")

            # Return to original directory
            os.chdir(original_dir)
        except Exception as e:
            print(f"Warning: Error during RDF-store cleanup: {e}")


def _wait_for_graphdb(timeout=180):
    """Wait for GraphDB to be ready to accept connections."""
    print("Waiting for GraphDB to start...")
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"GraphDB didn't start within {timeout} seconds")

        try:
            response = requests.get("http://localhost:7200/rest/repositories")
            if response.status_code == 200:
                print(
                    f"GraphDB is up and running! (took {time.time() - start_time:.1f} seconds)"
                )
                time.sleep(5)  # Additional safety margin
                return
        except requests.exceptions.ConnectionError:
            pass

        print(
            f"Still waiting for GraphDB... ({int(time.time() - start_time)}s elapsed)"
        )
        time.sleep(5)


def _get_or_create_repository():
    """Get the first available repository or create a new one."""
    response = requests.get("http://localhost:7200/rest/repositories")
    if not response.ok:
        raise RuntimeError(f"Failed to get list of repositories: {response.text}")

    repositories = response.json()
    if not repositories:
        print("No repositories found. Creating a test repository...")
        return _create_repository()

    # List all available repositories
    print("Available repositories:")
    for repo in repositories:
        repo_id = repo.get("id")
        repo_title = repo.get("title", "No title")
        print(f"  - {repo_id}: {repo_title}")

    # Use the first repository
    repo_id = repositories[0].get("id")
    print(f"Using repository: {repo_id}")
    return repo_id


def _create_repository():
    """Create a new test repository."""
    config = {"id": "test-repo", "title": "Test Repository", "type": "free"}

    response = requests.post("http://localhost:7200/rest/repositories", json=config)

    if not response.ok:
        raise RuntimeError(f"Failed to create repository: {response.text}")

    print("Repository 'test-repo' created successfully")
    return "test-repo"


def _load_test_data(repository_id):
    """Load test data files from tests/data into GraphDB."""
    tests_directory = Path(__file__).parent
    data_directory = tests_directory / "data"
    data_filenames = ["ontology.ttl", "data.ttl", "annotation.ttl"]

    # Get repository endpoint
    repo_endpoint = f"http://localhost:7200/repositories/{repository_id}/"

    # Find and load each file
    files_loaded = False
    for filename in data_filenames:
        file_path = data_directory / filename

        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping...")
            continue

        print(f"Loading {file_path}...")

        content_type = {
            ".owl": "application/rdf+xml",
            ".ttl": "text/turtle",
            ".nt": "application/n-triples",
            ".nq": "application/n-quads",
            ".trig": "application/trig",
            ".jsonld": "application/ld+json",
        }.get(file_path.suffix, "text/turtle")

        try:
            # Use requests to upload the file instead of curl subprocess
            with open(file_path, 'rb') as f:
                data = f.read()

            headers = {
                "Content-Type": "application/x-turtle"
            }

            graph_url = f"{repo_endpoint}rdf-graphs/service?graph=http://{filename.replace('.ttl', '')}.local/"

            response = requests.post(graph_url, data=data, headers=headers)

            if response.ok:
                print(f"Successfully loaded {file_path}")
                files_loaded = True
            else:
                print(f"Failed to load {file_path}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    if files_loaded:
        print("Test data loading completed")
    else:
        print("Warning: No test data files were loaded successfully")


@pytest.fixture(scope="session")
def algorithm_image(docker_client):
    """Build the algorithm Docker image for the entire test session. Depends on Flyover setup."""

    # Use mock algorithm located in /tests/mock_algorithm
    mock_algorithm_path = Path(__file__).parent / "mock_algorithm" / "v6-rdf-mock"

    if not mock_algorithm_path.exists():
        pytest.skip(f"Mock algorithm directory not found at {mock_algorithm_path}")

    # Use v6-rdf-mock as package name
    pkg_name = "v6-rdf-mock"

    # Create image tag from package name
    image_tag = f"{pkg_name}:ci-test"

    try:
        print(f"Building algorithm image from {mock_algorithm_path}...")
        print(f"Package name: {pkg_name}")
        print(f"Image tag: {image_tag}")

        # Copy v6-tools-rdf source to the mock algorithm directory temporarily
        v6_tools_src = Path(__file__).parent.parent / "src"
        v6_tools_pyproject = Path(__file__).parent.parent / "pyproject.toml"
        v6_tools_readme = Path(__file__).parent.parent / "README.md"

        temp_v6_dir = mock_algorithm_path / "v6-tools-rdf-src"
        if temp_v6_dir.exists():
            shutil.rmtree(temp_v6_dir)
        temp_v6_dir.mkdir()

        # Copy the source files
        shutil.copytree(v6_tools_src, temp_v6_dir / "src")
        shutil.copy2(v6_tools_pyproject, temp_v6_dir)
        shutil.copy2(v6_tools_readme, temp_v6_dir)

        try:
            # Build Docker image for the mock algorithm
            build_result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    image_tag,
                    "--build-arg",
                    f"PKG_NAME={pkg_name}",
                    str(mock_algorithm_path),
                ],
                check=True,
                timeout=300,
                capture_output=True,
                text=True,
            )

            if build_result.returncode != 0:
                pytest.skip(
                    f"Algorithm image build failed with exit code {build_result.returncode}:\n"
                    f"STDOUT: {build_result.stdout}\n"
                    f"STDERR: {build_result.stderr}"
                )

        finally:
            # Clean up temporary directory
            if temp_v6_dir.exists():
                shutil.rmtree(temp_v6_dir)

        # Verify the image was created
        try:
            inspect_result = subprocess.run(
                ["docker", "inspect", image_tag],
                check=True,
                capture_output=True,
                text=True,
            )

            image_info = json.loads(inspect_result.stdout)
            if not image_info or not image_info[0].get("Id"):
                pytest.skip(f"Built image {image_tag} has no valid ID")

            print(f"Successfully built algorithm image: {image_tag}")
            return {
                "tag": image_tag,
                "pkg_name": pkg_name,
                "id": image_info[0]["Id"],
                "info": image_info[0],
            }

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            pytest.skip(f"Failed to inspect built image {image_tag}: {e}")

    except subprocess.CalledProcessError as e:
        error_msg = f"Algorithm build failed: {e}"
        if e.stdout:
            error_msg += f"\nSTDOUT: {e.stdout}"
        if e.stderr:
            error_msg += f"\nSTDERR: {e.stderr}"
        pytest.skip(error_msg)
    except subprocess.TimeoutExpired:
        pytest.skip("Timeout building algorithm image (300s)")
    except Exception as e:
        pytest.skip(f"Unexpected error building algorithm image: {e}")


@pytest.fixture(scope="session")
def extra_node_config_file(algorithm_image):
    """Create a temporary YAML file with the allowed algorithms policy."""
    algorithm_name = algorithm_image["tag"]

    # Retrieve the node configuration
    tests_directory = Path(__file__).parent
    config_file_path = os.path.join(
        tests_directory, "data", "additional_vantage6_node_config.yaml"
    )

    # Read existing content before modification
    original_content = ""
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as f:
            original_content = f.read()

    # Append new content
    new_content = (
        f"\n"
        f"policies:\n"
        f"  allowed_algorithms:\n"
        f"    # Developer network start-up test\n"
        f"    - ^hello-world$\n"
        f"    # {algorithm_name}\n"
        f"    - ^{algorithm_name}$\n"
        f"  require_algorithm_pull: false\n"
    )

    with open(config_file_path, "a") as config_file:
        config_file.write(new_content)

    # Use the absolute path to ensure no issues
    full_path = os.path.abspath(config_file_path)

    print(f"Updated extra node config file: {full_path}")
    print(f"Allowing algorithm: {algorithm_name}")

    yield full_path

    # Restore original content (removing what we added)
    try:
        if original_content:
            # Restore to its original state
            with open(config_file_path, "w") as f:
                f.write(original_content)
            print(f"Restored original content of config file: {full_path}")
        else:
            # File didn't exist before, strip the content we added
            with open(config_file_path, "r") as f:
                current_content = f.read()
            # Remove the new content we appended
            restored_content = current_content.replace(new_content, "")
            with open(config_file_path, "w") as f:
                f.write(restored_content)
            print(f"Stripped added content from config file: {full_path}")
    except OSError as e:
        print(f"Warning: Could not restore config file {full_path}: {e}")


@pytest.fixture(scope="session")
def vantage6_network_session(docker_client, extra_node_config_file):
    """Set up the Vantage6 developer network for the entire test session."""
    from tests.integration.test_vantage6_integration import cleanup_vantage6_network

    network_info = {"status": "not_started", "created_containers": set()}

    # Check if vantage6 CLI is available
    try:
        result = subprocess.run(
            ["v6", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            pytest.skip(
                f"Vantage6 CLI not available (exit code {result.returncode}):\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        pytest.skip("Vantage6 CLI check timed out (10s)")
    except FileNotFoundError:
        pytest.skip(
            "Vantage6 CLI not found in PATH. Install with: pip install vantage6"
        )

    try:
        # Clean-up of any existing network first
        print("Cleaning up any existing Vantage6 networks...")
        cleanup_vantage6_network(
            {"created_containers": set()}, docker_client, force_remove_existing=True
        )
        time.sleep(5)

        # Capture containers before creating the network
        containers_before = set(
            container.id for container in docker_client.containers.list(all=True)
        )

        # Get test data directory
        tests_directory = Path(__file__).parent
        data_directory = tests_directory / "data"

        # Define config file paths
        server_config_path = data_directory / "additional_vantage6_server_config.yaml"
        store_config_path = data_directory / "additional_vantage6_store_config.yaml"

        # Define dataset paths and names
        datasets = [
            ("rdf_store", data_directory / "rdf_store.csv")
        ]

        # Create and start a demo network with extra node config
        print("Creating demo network: algorithm-ci-test.")
        print(f"Using IP: '{docker_client.docker_host}', as server IP.")
        create_args = [
            "v6",
            "dev",
            "create-demo-network",
            "--name",
            "algorithm-ci-test",
            "--server-url",
            str(docker_client.docker_host),
            "--extra-node-config",
            str(extra_node_config_file),
            "--extra-server-config",
            str(server_config_path),
            "--extra-store-config",
            str(store_config_path),
        ]

        # Add datasets dynamically
        for dataset_name, dataset_path in datasets:
            if dataset_path.exists():
                create_args.extend(["--add-dataset", dataset_name, str(dataset_path)])
            else:
                print(f"Warning: Dataset file {dataset_path} not found, skipping...")

        create_result = subprocess.run(
            create_args, timeout=300, capture_output=True, text=True
        )

        if create_result.returncode != 0:
            error_msg = f"Failed to create demo network (exit code {create_result.returncode}):\n"
            error_msg += f"STDOUT: {create_result.stdout}\n"
            error_msg += f"STDERR: {create_result.stderr}"
            pytest.skip(error_msg)

        print("Starting demo network...")
        start_result = subprocess.run(
            ["v6", "dev", "start-demo-network", "--name", "algorithm-ci-test"],
            timeout=300,
            capture_output=True,
            text=True,
        )

        if start_result.returncode != 0:
            error_msg = (
                f"Failed to start demo network (exit code {start_result.returncode}):\n"
            )
            error_msg += f"STDOUT: {start_result.stdout}\n"
            error_msg += f"STDERR: {start_result.stderr}"
            pytest.skip(error_msg)

        # Wait for the network to be ready
        print("Waiting for network to start...")
        max_wait = 90
        wait_interval = 5
        stable_count = 0
        required_stable_checks = 3

        for elapsed in range(0, max_wait, wait_interval):
            time.sleep(wait_interval)
            containers_after = set(
                container.id for container in docker_client.containers.list(all=True)
            )
            new_containers = containers_after - containers_before

            if len(new_containers) >= 4:
                service_containers = 0
                for container_id in new_containers:
                    try:
                        container = docker_client.containers.get(container_id)
                        if (
                            container.status == "running"
                            and "-run-" not in container.name
                            and "algorithm-store" not in container.name
                        ):
                            service_containers += 1
                    except docker.errors.NotFound:
                        pass

                if service_containers >= 3:
                    stable_count += 1
                    if stable_count >= required_stable_checks:
                        print(
                            f"Network stable and ready after {elapsed + wait_interval} seconds"
                        )
                        break
                else:
                    stable_count = 0

        containers_after = set(
            container.id for container in docker_client.containers.list(all=True)
        )
        network_info["created_containers"] = containers_after - containers_before
        network_info["status"] = "running"

        if len(network_info["created_containers"]) < 4:
            pytest.skip(
                "Network setup incomplete: "
                f"only {len(network_info['created_containers'])} containers created (expected ≥4)"
            )

        print(
            f"Network started with {len(network_info['created_containers'])} new containers"
        )

        yield network_info

    except subprocess.TimeoutExpired as e:
        error_msg = f"Timeout setting up Vantage6 demo network: {e}"
        pytest.skip(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error setting up Vantage6 demo network: {e}"
        pytest.skip(error_msg)

    finally:
        # Clean-up after all tests are done
        print("Cleaning up Vantage6 network...")
        cleanup_vantage6_network(network_info, docker_client)


@pytest.fixture(scope="session")
def authentication(vantage6_network_session, docker_client) -> Client:
    """
    This function authenticates a client.

    It creates a client with the given configuration details, authenticates the client,
    and sets up encryption for the client.

    Parameters:
    vantage6_network_session: The running vantage6 network session (dependency)

    Returns:
        Client: An authenticated client with encryption set up.
    """
    # Ensure the network is running before attempting authentication
    if vantage6_network_session["status"] != "running":
        pytest.skip("Vantage6 network is not running - cannot authenticate")

    # Additional wait to ensure all services are fully ready
    print("Waiting for network services to be fully ready for authentication...")
    time.sleep(10)

    vantage6_config = {
        "server_url": "http://localhost",
        "server_port": 7601,
        "server_api": "/api",
        "username": "dev_admin",
        "password": "password",
    }

    # Create a client
    vantage6_client = Client(
        vantage6_config.get("server_url"),
        vantage6_config.get("server_port"),
        vantage6_config.get("server_api"),
        log_level="debug",
    )

    # Authenticate the client with a retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            vantage6_client.authenticate(
                vantage6_config.get("username"), vantage6_config.get("password")
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                pytest.skip(f"Failed to authenticate after {max_retries} attempts: {e}")
            print(
                f"Authentication attempt {attempt + 1} failed, retrying... Error: {e}"
            )
            time.sleep(5)

    # Verify authentication
    assert hasattr(vantage6_client, "token"), "Client should have a token attribute"
    assert vantage6_client.token, "Authentication token should be present and not empty"

    return vantage6_client


@pytest.fixture(scope="session")
def algorithm_image_name(algorithm_image):
    """Get the algorithm image name for testing."""
    return algorithm_image["tag"]


def get_docker_host():
    """
    Determine the appropriate Docker host address based on the operating system.

    Returns:
        str: The Docker host address to use
    """
    system = platform.system().lower()

    if system in ["windows", "darwin"]:  # Windows or macOS
        return "http://host.docker.internal"
    else:  # Linux and others
        try:
            # Try to get the Docker bridge IP
            result = subprocess.run(
                ["docker", "network", "inspect", "bridge"],
                capture_output=True,
                text=True,
                check=True,
            )
            import json

            bridge_info = json.loads(result.stdout)
            gateway = bridge_info[0]["IPAM"]["Config"][0]["Gateway"]

            # The gateway might not have the http scheme included
            if not gateway.startswith("http"):
                gateway = f"http://{gateway}"
            return gateway
        except (
            subprocess.CalledProcessError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ):
            # Assume it is a default Docker IP if inspection fails
            return "http://172.17.0.1"


# Pytest markers for organising tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for algorithm functions")
    config.addinivalue_line("markers", "integration: Integration tests with Vantage6")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line(
        "markers", "vantage6: Tests that require Vantage6 infrastructure"
    )
    config.addinivalue_line("markers", "docker: Tests that require Docker")


def pytest_collection_modifyitems(config, items):
    """Modify the test collection to add markers based on test names/paths."""
    for item in items:
        # Add slow marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Add a vantage6 marker to tests that use vantage6
        if "vantage6" in item.nodeid or "vantage6" in item.name:
            item.add_marker(pytest.mark.vantage6)

        # Add docker marker to tests that use docker
        if "docker" in item.nodeid or "docker" in item.name:
            item.add_marker(pytest.mark.docker)
