from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.embeddings.voyageai.base import CONTEXT_MODELS


def test_embedding_class():
    emb = VoyageEmbedding(model_name="", voyage_api_key="NOT_A_VALID_KEY")
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 7
    assert emb.model_name == ""


def test_embedding_class_voyage_2():
    emb = VoyageEmbedding(
        model_name="voyage-2", voyage_api_key="NOT_A_VALID_KEY", truncation=True
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 72
    assert emb.model_name == "voyage-2"
    assert emb.truncation
    assert emb.output_dimension is None
    assert emb.output_dtype is None


def test_embedding_class_voyage_2_with_batch_size():
    emb = VoyageEmbedding(
        model_name="voyage-2", voyage_api_key="NOT_A_VALID_KEY", embed_batch_size=49
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 49
    assert emb.model_name == "voyage-2"
    assert emb.truncation is None
    assert emb.output_dimension is None
    assert emb.output_dtype is None


def test_embedding_class_voyage_3_large_with_output_dimension():
    emb = VoyageEmbedding(
        model_name="voyage-3-large",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dimension=512,
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 7
    assert emb.model_name == "voyage-3-large"
    assert emb.truncation is None
    assert emb.output_dimension == 512
    assert emb.output_dtype is None


def test_embedding_class_voyage_3_large_with_output_dtype():
    emb = VoyageEmbedding(
        model_name="voyage-3-large",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dtype="float",
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 7
    assert emb.model_name == "voyage-3-large"
    assert emb.truncation is None
    assert emb.output_dimension is None
    assert emb.output_dtype == "float"


def test_voyageai_embedding_class():
    names_of_base_classes = [b.__name__ for b in VoyageEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_embedding_class_context_model():
    emb = VoyageEmbedding(
        model_name="voyage-context-3",
        voyage_api_key="NOT_A_VALID_KEY",
        output_dimension=1024
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 7
    assert emb.model_name == "voyage-context-3"
    assert emb.output_dimension == 1024
    assert emb.output_dtype is None


def test_embedding_class_context_model_with_params():
    emb = VoyageEmbedding(
        model_name="voyage-context-3",
        voyage_api_key="NOT_A_VALID_KEY",
        truncation=True,
        output_dtype="float",
        output_dimension=512
    )
    assert isinstance(emb, BaseEmbedding)
    assert emb.embed_batch_size == 7
    assert emb.model_name == "voyage-context-3"
    assert emb.output_dimension == 512
    assert emb.output_dtype == "float"


def test_context_model_detection():
    context_emb = VoyageEmbedding(
        model_name="voyage-context-3", voyage_api_key="NOT_A_VALID_KEY"
    )
    regular_emb = VoyageEmbedding(
        model_name="voyage-3", voyage_api_key="NOT_A_VALID_KEY"
    )

    assert context_emb.model_name in CONTEXT_MODELS
    assert regular_emb.model_name not in CONTEXT_MODELS
