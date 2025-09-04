"""
Prepare the docs folder for building the documentation.

This file will:
1. Update the mkdocs.yml file to include all example notebooks
2. Create API reference pages for all integration folders
3. Update the search paths for the mkdocstrings plugin
4. Copy over the latest CHANGELOG.md/CONTRIBUTING.md/DOCS_README.md
   to the docs/docs folder
"""

import os

import toml
import yaml

MKDOCS_YML = "mkdocs.yml"

# examples config
EXAMPLES_DIR = "docs/examples"
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

    # get all example notebooks
    notebooks = []
    for root, dirs, files in os.walk(EXAMPLES_DIR):
        for file in files:
            if file.endswith(".ipynb"):
                notebooks.append(os.path.join(root, file))

    # update the mkdocs.yml nav section
    examples_idx = -1
    for idx, item in enumerate(mkdocs["nav"]):
        if "Examples" in item:
            examples_idx = idx
            break

    for path_name, label in FOLDER_NAME_TO_LABEL.items():
        path_name = os.path.join(
            EXAMPLES_DIR.replace("examples", ""), path_name.replace("./", "")
        )

        label_idx = -1
        for idx, item in enumerate(mkdocs["nav"][examples_idx]["Examples"]):
            if label in item:
                label_idx = idx
                break

        # Add or clear the label
        if label_idx == -1:
            mkdocs["nav"][examples_idx]["Examples"].append({label: []})
        else:
            mkdocs["nav"][examples_idx]["Examples"][label_idx][label] = []

        for file_name in os.listdir(path_name):
            if file_name.endswith(".ipynb"):
                toc_path_name = "./" + os.path.join(
                    path_name.replace("docs/", ""), file_name
                )
                if (
                    toc_path_name
                    not in mkdocs["nav"][examples_idx]["Examples"][label_idx][
                        label
                    ]
                ):
                    mkdocs["nav"][examples_idx]["Examples"][label_idx][
                        label
                    ].append(toc_path_name)
            if os.path.isdir(os.path.join(path_name, file_name)):
                for root, dirs, files in os.walk(
                    os.path.join(path_name, file_name)
                ):
                    for file in files:
                        if file.endswith(".ipynb"):
                            toc_path_name = "./" + os.path.join(
                                root.replace("docs/", ""), file
                            )
                            if (
                                toc_path_name
                                not in mkdocs["nav"][examples_idx]["Examples"][
                                    label_idx
                                ][label]
                            ):
                                mkdocs["nav"][examples_idx]["Examples"][
                                    label_idx
                                ][label].append(toc_path_name)

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
                    search_paths.append(root)
                    # special cases
                    if folder_name == "vector_stores":
                        folder_name = "storage/vector_store"
                    elif folder_name == "indices/managed":
                        folder_name = "indices"
                    elif folder_name == "graph_stores":
                        folder_name = "storage/graph_stores"

                    full_path = os.path.join("docs/api_reference", folder_name)
                    module_name = import_path.split(".")[-1] + ".md"
                    os.makedirs(full_path, exist_ok=True)
                    with open(os.path.join(full_path, module_name), "w") as f:
                        f.write(api_ref)

                    # update the mkdocs.yml nav section
                    api_ref_idx = -1
                    for idx, item in enumerate(mkdocs["nav"]):
                        if "API Reference" in item:
                            api_ref_idx = idx
                            break

                    if "storage" in folder_name:
                        label = "Storage"
                    else:
                        label = INTEGRATION_FOLDER_TO_LABEL[
                            import_path.split(".")[1]
                        ]

                    label_idx = -1
                    for idx, item in enumerate(
                        mkdocs["nav"][api_ref_idx]["API Reference"]
                    ):
                        if label in item:
                            label_idx = idx
                            break

                    if label_idx == -1:
                        mkdocs["nav"][api_ref_idx]["API Reference"].append(
                            {label: []}
                        )

                    toc_path_name = "./" + os.path.join(
                        "api_reference", folder_name, module_name
                    )
                    if (
                        toc_path_name
                        not in mkdocs["nav"][api_ref_idx]["API Reference"][
                            label_idx
                        ][label]
                    ):
                        # storage is a special case, multi-level
                        if label == "Storage":
                            sub_path = folder_name.split("/")[-1]
                            sub_label = sub_path.replace("_", " ").title()
                            sub_label_idx = -1
                            for (
                                existing_sub_label_idx,
                                existing_sub_label,
                            ) in enumerate(
                                mkdocs["nav"][api_ref_idx]["API Reference"][
                                    label_idx
                                ][label]
                            ):
                                if sub_label in existing_sub_label:
                                    sub_label_idx = existing_sub_label_idx
                                    break

                            if sub_label_idx == -1:
                                mkdocs["nav"][api_ref_idx]["API Reference"][
                                    label_idx
                                ][label].append({sub_label: []})

                            if (
                                toc_path_name
                                not in mkdocs["nav"][api_ref_idx][
                                    "API Reference"
                                ][label_idx][label][sub_label_idx][sub_label]
                            ):
                                mkdocs["nav"][api_ref_idx]["API Reference"][
                                    label_idx
                                ][label][sub_label_idx][sub_label].append(
                                    toc_path_name
                                )

                            # sort per sub-label
                            mkdocs["nav"][api_ref_idx]["API Reference"][
                                label_idx
                            ][label][sub_label_idx][sub_label] = sorted(
                                mkdocs["nav"][api_ref_idx]["API Reference"][
                                    label_idx
                                ][label][sub_label_idx][sub_label],
                                key=lambda x: next(iter(x.keys()))
                                if isinstance(x, dict)
                                else x,
                            )
                        else:
                            mkdocs["nav"][api_ref_idx]["API Reference"][
                                label_idx
                            ][label].append(toc_path_name)

                    # maintain sorting per label
                    mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                        label
                    ] = sorted(
                        mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                            label
                        ],
                        key=lambda x: next(iter(x.keys()))
                        if isinstance(x, dict)
                        else x,
                    )

    # add existing api reference pages to nav
    api_ref_idx = -1
    for idx, item in enumerate(mkdocs["nav"]):
        if "API Reference" in item:
            api_ref_idx = idx
            break

    for root, _, files in os.walk("docs/api_reference"):
        for file in files:
            if file.endswith(".md"):
                toc_path_name = os.path.join(
                    root.replace("docs/api_reference", "./api_reference"),
                    file,
                )

                if toc_path_name == "./api_reference/index.md":
                    continue

                if "storage" in root:
                    label = "Storage"
                else:
                    try:
                        label = INTEGRATION_FOLDER_TO_LABEL[
                            root.split("/")[-1]
                        ]
                    except KeyError:
                        # Safe net to avoid blocking the build for some misconfiguration
                        print(
                            f"Unable to find {root.split('/')[-1]} in INTEGRATION_FOLDER_TO_LABEL"
                        )
                        continue

                label_idx = -1
                for idx, item in enumerate(
                    mkdocs["nav"][api_ref_idx]["API Reference"]
                ):
                    if label in item:
                        label_idx = idx
                        break

                if label_idx == -1:
                    mkdocs["nav"][api_ref_idx]["API Reference"].append(
                        {label: []}
                    )

                if "storage" in root:
                    sub_path = root.split("/")[-1]
                    sub_label = sub_path.replace("_", " ").title()
                    sub_label_idx = -1
                    for (
                        existing_sub_label_idx,
                        existing_sub_label,
                    ) in enumerate(
                        mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                            label
                        ]
                    ):
                        if sub_label in existing_sub_label:
                            sub_label_idx = existing_sub_label_idx
                            break

                    if sub_label_idx == -1:
                        mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                            label
                        ].append({sub_label: []})

                    if (
                        toc_path_name
                        not in mkdocs["nav"][api_ref_idx]["API Reference"][
                            label_idx
                        ][label][sub_label_idx][sub_label]
                    ):
                        mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                            label
                        ][sub_label_idx][sub_label].append(toc_path_name)

                    # sort per sub-label
                    mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                        label
                    ][sub_label_idx][sub_label] = sorted(
                        mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                            label
                        ][sub_label_idx][sub_label],
                        key=lambda x: next(iter(x.keys()))
                        if isinstance(x, dict)
                        else x,
                    )
                elif (
                    toc_path_name
                    not in mkdocs["nav"][api_ref_idx]["API Reference"][
                        label_idx
                    ][label]
                ):
                    mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                        label
                    ].append(toc_path_name)

                    # sort per label
                    mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                        label
                    ] = sorted(
                        mkdocs["nav"][api_ref_idx]["API Reference"][label_idx][
                            label
                        ],
                        key=lambda x: next(iter(x.keys()))
                        if isinstance(x, dict)
                        else x,
                    )

    # sort the API Reference nav section
    mkdocs["nav"][api_ref_idx]["API Reference"] = sorted(
        mkdocs["nav"][api_ref_idx]["API Reference"],
        key=lambda x: next(iter(x.keys())) if isinstance(x, dict) else x,
    )

    # sort the examples
    for idx, item in enumerate(mkdocs["nav"][examples_idx]["Examples"]):
        if isinstance(item, dict):
            for key in item:
                mkdocs["nav"][examples_idx]["Examples"][idx][key] = sorted(
                    mkdocs["nav"][examples_idx]["Examples"][idx][key],
                    key=lambda x: next(iter(x.keys()))
                    if isinstance(x, dict)
                    else x,
                )

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
    os.system("cp ../CHANGELOG.md ./docs/CHANGELOG.md")
    os.system("cp ../CONTRIBUTING.md ./docs/CONTRIBUTING.md")
    os.system("cp ./README.md ./docs/DOCS_README.md")


if __name__ == "__main__":
    main()
