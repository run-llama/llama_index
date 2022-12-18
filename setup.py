"""Set up the package."""
import sys
from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).absolute().parents[0] / "gpt_index" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "langchain",
    "openai",
    "dataclasses_json",
    "transformers",
    "nltk",
    # for openAI embeddings
    "matplotlib",
    "plotly",
    "scipy",
    "scikit-learn",
]

# NOTE: if python version >= 3.9, install tiktoken
if sys.version_info >= (3, 9):
    install_requires.extend(["tiktoken"])

setup(
    name="gpt_index",
    version=__version__,
    packages=find_packages(),
    description="Building an index of GPT summaries.",
    install_requires=install_requires,
    long_description=long_description,
    license="MIT",
    url="https://github.com/jerryjliu/gpt_index",
    include_package_data=True,
    long_description_content_type="text/markdown",
)
