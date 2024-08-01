from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding


def test_openvinoembedding_class():
    names_of_base_classes = [b.__name__ for b in OpenVINOEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_openvinoembedding_get_text_embedding(tmp_path):
    model_dir = str(tmp_path / "models/bge_ov")
    OpenVINOEmbedding.create_and_save_openvino_model(
        "BAAI/bge-small-en-v1.5", model_dir
    )
    embed_model = OpenVINOEmbedding(model_id_or_path=model_dir)
    embeddings = embed_model.get_text_embedding("Hello World!")

    assert len(embeddings) == 384
