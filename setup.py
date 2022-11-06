"""Set up the package."""
from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).absolute().parents[0] / "gpt_db_retrieve" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="gpt_db_retrieve",
    version=__version__,
    packages=find_packages(),
    description="Building an index of GPT summaries.",
    install_requires=["langchain", "openai", "dataclasses_json", "transformers"],
    long_description=long_description,
    license="MIT",
    url="https://github.com/jerryjliu/gpt_db_retrieve",
    include_package_data=True,
    long_description_content_type="text/markdown",
)
