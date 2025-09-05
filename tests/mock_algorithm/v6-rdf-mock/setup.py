from os import path
from codecs import open
from setuptools import setup, find_packages

# we're using a README.md, if you do not have this in your folder, simply
# replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Here you specify the meta-data of your package. The `name` argument is
# needed in some other steps.
setup(
    name="v6-rdf-mock",
    version="1.0.0",
    description="A very basic and not privacy enhancing mock algorithm that checks whether the RDF-endpoint is accessible and the vantage6_strongaya_rdf tools are working. Stripped down algorithm, not to be used elsewhere.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vantage6-algorithm-tools",
        "pandas",
        "vantage6-strongaya-general @ git+https://github.com/STRONGAYA/v6-tools-general.git@v1.0.2",
    ],
)
