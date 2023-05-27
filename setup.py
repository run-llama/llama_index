"""Set up the package."""
import sys
from pathlib import Path
import os
from setuptools import find_packages, setup

DEFAULT_PACKAGE_NAME = "llama_index"
PACKAGE_NAME = os.environ.get("PACKAGE_NAME_OVERRIDE", DEFAULT_PACKAGE_NAME)

with open(Path(__file__).absolute().parents[0] / "llama_index" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "dataclasses_json",
    "langchain>=0.0.154",
    "sqlalchemy>=2.0.15",
    "numpy",
    "tenacity>=8.2.0,<9.0.0",
    "openai>=0.26.4",
    "pandas",
    "urllib3<2",
    "fsspec>=2023.5.0",
    "typing-inspect==0.8.0",
    "typing_extensions==4.5.0",
]

# NOTE: if python version >= 3.9, install tiktoken
# else install transformers
if sys.version_info >= (3, 9):
    install_requires.extend(["tiktoken"])
else:
    install_requires.extend(["transformers"])

setup(
    author="Jerry Liu",
    name=PACKAGE_NAME,
    version=__version__,
    packages=find_packages(),
    description="Interface between LLMs and your data",
    install_requires=install_requires,
    long_description=long_description,
    license="MIT",
    url="https://github.com/jerryjliu/llama_index",
    include_package_data=True,
    long_description_content_type="text/markdown",
)
