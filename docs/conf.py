"""Configuration for sphinx."""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../"))

with open("../llama_index/VERSION") as f:
    version = f.read()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "LlamaIndex 🦙"
copyright = "2022, Jerry Liu"
author = "Jerry Liu"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "m2r2",
    "myst_nb",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_reredirects",
]

myst_heading_anchors = 4
# TODO: Fix the non-consecutive header level in our docs, until then
# disable the sphinx/myst warnings
suppress_warnings = ["myst.header"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store","DOCS_README.md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = project + " " + version
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
    "css/algolia.css",
    "https://cdn.jsdelivr.net/npm/@docsearch/css@3",
]
html_js_files = [
    "js/mendablesearch.js",
    (
        "https://cdn.jsdelivr.net/npm/@docsearch/js@3.3.3/dist/umd/index.js",
        {"defer": "defer"},
    ),
    ("js/algolia.js", {"defer": "defer"}),
]

nb_execution_mode = "off"
autodoc_pydantic_model_show_json_error_strategy = "coerce"
nitpicky = True

## Redirects

redirects = {
    "end_to_end_tutorials/usage_pattern": "/understanding/understanding.html",
    "end_to_end_tutorials/one_click_observability": "/module_guides/observability/observability.html",
    "end_to_end_tutorials/dev_practices/production_rag": "/optimizing/production_rag.html",
    "end_to_end_tutorials/dev_practices/evaluation": "/optimizing/evaluation/evaluation.html",
    "end_to_end_tutorials/discover_llamaindex": "/getting_started/discover_llamaindex.html",
    "end_to_end_tutorials/finetuning": "/optimizing/fine-tuning/fine-tuning.html",
    "end_to_end_tutorials/low_level/root": "/optimizing/building_rag_from_scratch.html",
    "end_to_end_tutorials/use_cases": "/use_cases/q_and_a.html",
    "core_modules/data_modules/connector": "/module_guides/loading/connector/root.html",
    "core_modules/data_modules/documents_and_nodes/root": "/module_guides/loading/documents_and_nodes/root.html",
    "core_modules/data_modules/node_parsers/root": "/module_guides/loading/node_parsers/root.html",
    "core_modules/data_modules/storage/root": "/module_guides/storing/storing.html",
    "core_modules/data_modules/index/root": "/module_guides/indexing/indexing.html",
    "core_modules/query_modules/query_engine/root": "/module_guides/deploying/query_engine/root.html",
    "core_modules/query_modules/chat_engines/root": "/module_guides/deploying/chat_engines/root.html",
    "core_modules/query_modules/retriever/root": "/module_guides/querying/retriever/root.html",
    "core_modules/query_modules/router/root": "/module_guides/querying/router/root.html",
    "core_modules/query_modules/node_postprocessors/root": "/module_guides/querying/node_postprocessors/root.html",
    "core_modules/query_modules/response_synthesizers/root": "/module_guides/querying/response_synthesizers/root.html",
    "core_modules/query_modules/structured_outputs/root": "/optimizing/advanced_retrieval/structured_outputs/structured_outputs.html",
    "core_modules/agent_modules/agents/root": "/module_guides/deploying/agents/root.html",
    "core_modules/agent_modules/tools/root": "/module_guides/deploying/agents/tools/root.html",
    "core_modules/model_modules/llms/root": "/module_guides/models/llms.html",
    "core_modules/model_modules/embeddings/root": "/module_guides/models/embeddings.html",
    "core_modules/model_modules/prompts": "/module_guides/models/prompts.html",
    "core_modules/supporting_modules/service_context": "/module_guides/supporting_modules/service_context.html",
    "core_modules/supporting_modules/callbacks/root": "/module_guides/observability/callbacks/root.html",
    "core_modules/supporting_modules/evaluation/root": "/module_guides/evaluating/root.html",
    "core_modules/supporting_modules/cost_analysis/root": "/understanding/evaluating/cost_analysis/root.html",    
}
