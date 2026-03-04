# /// script
# dependencies = [
#   "toml",
#   "pyaml",
# ]
# ///

import os

import toml
import yaml

MKDOCS_YML = "./api_reference/mkdocs.yml"

# examples config
EXAMPLES_DIR = "./examples"
FOLDER_NAME_TO_LABEL = {
    "./examples/agent": "Agents",
    "./examples/cookbooks": "Cookbooks",
    "./examples/chat_engine": "Chat Engines",
    "./examples/customization": "Customization",
    "./examples/data_connectors": "Data Connectors",
    "./examples/discover_llamaindex": "Discover LlamaIndex",
    "./examples/docstore": "Docstores",
    "./examples/embeddings": "Embeddings",
    "./examples/evaluation": "Evaluation",
    "./examples/finetuning": "Finetuning",
    "./examples/ingestion": "Ingestion",
    "./examples/llama_dataset": "Llama Datasets",
    "./examples/llama_hub": "Llama Hub",
    "./examples/llm": "LLMs",
    "./examples/low_level": "Low Level",
    "./examples/managed": "Managed Indexes",
    "./examples/memory": "Memory",
    "./examples/metadata_extraction": "Metadata Extractors",
    "./examples/multi_modal": "Multi-Modal",
    "./examples/multi_tenancy": "Multi-Tenancy",
    "./examples/node_parsers": "Node Parsers & Text Splitters",
    "./examples/node_postprocessor": "Node Postprocessors",
    "./examples/objects": "Object Stores",
    "./examples/observability": "Observability",
    "./examples/output_parsing": "Output Parsers",
    "./examples/param_optimizer": "Param Optimizer",
    "./examples/pipeline": "Query Pipeline",
    "./examples/prompts": "Prompts",
    "./examples/query_engine": "Query Engines",
    "./examples/query_transformations": "Query Transformations",
    "./examples/response_synthesizers": "Response Synthesizers",
    "./examples/retrievers": "Retrievers",
    "./examples/tools": "Tools",
    "./examples/transforms": "Transforms",
    "./examples/usecases": "Use Cases",
    "./examples/vector_stores": "Vector Stores",
    "./examples/workflow": "Workflow",
}

# integration config
INTEGRATION_FOLDERS = [
    # "../llama-index-networks",
    # "../llama-index-finetuning",
    "../llama-index-packs",
    "../llama-index-integrations",
    # "../llama-index-cli",
]
EXCLUDED_INTEGRATION_FOLDERS = [
    "llama-index-integrations/agent",
]
INTEGRATION_FOLDER_TO_LABEL = {
    "finetuning": "Fine-tuning",
    "llms": "LLMs",
    "agent": "Agents",
    "callbacks": "Callbacks",
    "chat_engines": "Chat Engines",
    "embeddings": "Embeddings",
    "evaluation": "Evaluation",
    "extractors": "Metadata Extractors",
    "graph_rag": "Graph RAG",
    "indices": "Indexes",
    "ingestion": "Ingestion",
    "instrumentation": "Instrumentation",
    "llama_dataset": "Llama Datasets",
    "packs": "Llama Packs",
    "memory": "Memory",
    "multi_modal_llms": "Multi-Modal LLMs",
    "node_parsers": "Node Parsers & Text Splitters",
    "node_parser": "Node Parsers & Text Splitters",
    "objects": "Object Stores",
    "observability": "Observability",
    "output_parsers": "Output Parsers",
    "postprocessor": "Node Postprocessors",
    "program": "Programs",
    "prompts": "Prompts",
    "query_engine": "Query Engines",
    "query_pipeline": "Query Pipeline",
    "question_gen": "Question Generators",
    "protocols": "Protocols",
    "readers": "Readers",
    "response_synthesizers": "Response Synthesizers",
    "retrievers": "Retrievers",
    "schema": "Schema",
    "selectors": "Selectors",
    "sparse_embeddings": "Sparse Embeddings",
    "storage": "Storage",
    "tools": "Tools",
    "workflow": "Workflow",
    "llama_deploy": "LlamaDeploy",
    "message_queues": "Message Queues",
    "voice_agents": "Voice Agents",
}
API_REF_TEMPLATE = """::: {import_path}
    options:
      members:
{members}
"""
API_REF_MEMBER_TEMPLATE = """        - {member}"""


def main():
    with open(MKDOCS_YML) as f:
        mkdocs = yaml.safe_load(f)

    # find all pyproject.toml files in the integration folders
    # each toml file has a toml['tool']['llamahub']['import_path'] key that we need
    # toml['tool']['llamahub']['class_authors'] contains a list of exposed classes
    # For each class, we need to create an API reference page
    search_paths = []
    for folder in INTEGRATION_FOLDERS:
        for root, dirs, files in os.walk(folder):
            if ".venv" in root:
                continue
            for file in files:
                # check if the current root is in the excluded integration folders
                if any(
                    excluded_folder in root
                    for excluded_folder in EXCLUDED_INTEGRATION_FOLDERS
                ):
                    continue

                if file == "pyproject.toml":
                    toml_path = os.path.join(root, file)
                    if ".venv" in toml_path:
                        continue

                    with open(toml_path) as f:
                        toml_data = toml.load(f)

                    import_path = toml_data["tool"]["llamahub"]["import_path"]
                    class_authors = toml_data["tool"]["llamahub"][
                        "class_authors"
                    ]
                    members = "\n".join(
                        [
                            API_REF_MEMBER_TEMPLATE.format(member=member)
                            for member in class_authors
                        ]
                    )
                    api_ref = API_REF_TEMPLATE.format(
                        import_path=import_path, members=members
                    )

                    folder_name = "/".join(import_path.split(".")[1:-1])
                    search_paths.append(os.path.join("../", root))
                    # special cases
                    if folder_name == "vector_stores":
                        folder_name = "storage/vector_store"
                    elif folder_name == "indices/managed":
                        folder_name = "indices"
                    elif folder_name == "graph_stores":
                        folder_name = "storage/graph_stores"

                    full_path = os.path.join(
                        "./api_reference/api_reference", folder_name
                    )
                    module_name = import_path.split(".")[-1] + ".md"
                    os.makedirs(full_path, exist_ok=True)
                    with open(os.path.join(full_path, module_name), "w") as f:
                        f.write(api_ref)

    # update search paths
    for i, plugin in enumerate(mkdocs["plugins"]):
        if "mkdocstrings" in plugin:
            for search_path in search_paths:
                if (
                    search_path
                    not in mkdocs["plugins"][i]["mkdocstrings"]["handlers"][
                        "python"
                    ]["paths"]
                ):
                    mkdocs["plugins"][i]["mkdocstrings"]["handlers"]["python"][
                        "paths"
                    ].append(search_path)

    # write the updated mkdocs.yml
    with open(MKDOCS_YML, "w") as f:
        yaml.dump(mkdocs, f)

    # copy over extra files
    os.system("cp ../CHANGELOG.md ./src/content/docs/framework/CHANGELOG.md")

    # Ensure CHANGELOG had the proper astro header
    changelog_contents = ""
    with open("./src/content/docs/framework/CHANGELOG.md", "r") as f:
        changelog_contents = f.read()

    astro_header = "---\ntitle: ChangeLog\n---"
    changelog_contents = changelog_contents.replace(
        "# ChangeLog\n", astro_header
    )

    with open("./src/content/docs/framework/CHANGELOG.md", "w") as f:
        f.write(changelog_contents)


if __name__ == "__main__":
    main()
