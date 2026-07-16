# Basic python3 image as base
FROM harbor2.vantage6.ai/infrastructure/algorithm-base

# This is a placeholder that should be overloaded by invoking docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="v6-rdf-mock"

# Install git and other dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

# Install federated algorithm first (this will install vantage6-strongaya-general from git)
COPY . /app
RUN pip install /app

# Copy the local v6-tools-rdf source code and install it
# Note: These files need to be copied into the build context
COPY v6-tools-rdf-src /tmp/v6-tools-rdf
RUN pip install /tmp/v6-tools-rdf

# Set environment variable to make name of the package available within the docker image.
ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `wrap_algorithm()` when the image is run.
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
