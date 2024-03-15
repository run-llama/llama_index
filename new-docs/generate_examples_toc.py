import os

folder_name_to_label = {
    "./examples/agent": "Agents",
    "./examples/callbacks": "Callbacks",
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
    "./examples/metadata_extraction": "Metadata Extractors",
    "./examples/multi_modal": "Multi-Modal",
    "./examples/multi_tenancy": "Multi-Tenancy",
    "./examples/node_parsers": "Node Parsers & Text Splitters",
    "./examples/node_postprocessor": "Node Postprocessors",
    "./examples/objects": "Object Stores",
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
}


def get_toc_from_folder(path_name):
    final_toc = ""
    for file_name in os.listdir(path_name):
        if file_name.endswith(".ipynb"):
            toc_path_name = os.path.join(
                path_name.replace("docs/", ""), file_name
            )
            final_toc += f"  - {toc_path_name}\n"
        if os.path.isdir(os.path.join(path_name, file_name)):
            final_toc += get_toc_from_folder(
                os.path.join(path_name, file_name)
            )
    return final_toc


final_toc = ""
for path_name, label in folder_name_to_label.items():
    path_name = os.path.join("./docs", path_name.replace("./", ""))
    final_toc += f"- {label}:\n"
    final_toc += get_toc_from_folder(path_name)

print(final_toc)
