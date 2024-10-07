import json
from pathlib import Path

from llama_index.core.schema import Document as LIDocument

from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.schema import BaseNode

ROOT_DIR_PATH = Path(__file__).resolve().parent


def _deterministic_id_func(i: int, doc: BaseNode) -> str:
    doc_dict = json.loads(doc.get_content())
    return f"{doc_dict['file-info']['document-hash']}_{i}"


def test_parse_nodes():
    with open(ROOT_DIR_PATH / "data" / "inp_li_doc.json") as f:
        data_json = f.read()
    li_doc = LIDocument.from_json(data_json)
    node_parser = DoclingNodeParser(
        id_func=_deterministic_id_func,
    )
    nodes = node_parser._parse_nodes(nodes=[li_doc])
    act_data = {"root": [n.model_dump() for n in nodes]}
    with open(ROOT_DIR_PATH / "data" / "out_parse_nodes.json") as f:
        exp_data = json.load(fp=f)
    assert act_data == exp_data


def test_get_nodes_from_docs():
    with open(ROOT_DIR_PATH / "data" / "inp_li_doc.json") as f:
        data_json = f.read()
    li_doc = LIDocument.from_json(data_json)
    node_parser = DoclingNodeParser(
        id_func=_deterministic_id_func,
    )
    nodes = node_parser.get_nodes_from_documents(documents=[li_doc])
    act_data = {"root": [n.model_dump() for n in nodes]}
    with open(ROOT_DIR_PATH / "data" / "out_get_nodes_from_docs.json") as f:
        exp_data = json.load(fp=f)
    assert act_data == exp_data
