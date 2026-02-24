"""Shim setup.py for backward compatibility with editable installs and legacy tooling."""

from setuptools import setup

setup(
    name="llama-index",
    version="0.14.8",
    description="Interface between LLMs and your data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="Jerry Liu",
    author_email="jerry@llamaindex.ai",
    url="https://llamaindex.ai",
    project_urls={
        "Documentation": "https://docs.llamaindex.ai/en/stable/",
        "Repository": "https://github.com/run-llama/llama_index",
    },
    python_requires=">=3.9,<4.0",
    keywords=["LLM", "NLP", "RAG", "data", "devtools", "index", "retrieval"],
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "llama-index-cli>=0.5.0,<0.6; python_version > '3.9'",
        "llama-index-core>=0.14.8,<0.15.0",
        "llama-index-embeddings-openai>=0.5.0,<0.6",
        "llama-index-indices-managed-llama-cloud>=0.4.0",
        "llama-index-llms-openai>=0.6.0,<0.7",
        "llama-index-readers-file>=0.5.0,<0.6",
        "llama-index-readers-llama-parse>=0.4.0",
        "nltk>3.8.1",
    ],
    entry_points={
        "console_scripts": [
            "llamaindex-cli=llama_index.cli.command_line:main",
        ],
    },
    package_dir={"llama_index": "_llama-index/llama_index"},
    packages=["llama_index"],
)
