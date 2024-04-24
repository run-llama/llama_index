import pytest
import inspect

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nvidia import NVIDIAEmbedding

from openai import AuthenticationError


def test_embedding_class():
    emb = NVIDIAEmbedding()
    assert isinstance(emb, BaseEmbedding)


def test_nvidia_embedding_param_setting():
    emb = NVIDIAEmbedding(
        model="test-model",
        truncate="END",
        timeout=20,
        max_retries=10,
        embed_batch_size=15,
    )

    assert emb.model == "test-model"
    assert emb.truncate == "END"
    assert emb._client.timeout == 20
    assert emb._client.max_retries == 10
    assert emb._aclient.timeout == 20
    assert emb._aclient.max_retries == 10
    assert emb.embed_batch_size == 15


def test_nvidia_embedding_throws_on_batches_larger_than_259():
    with pytest.raises(ValueError):
        NVIDIAEmbedding(embed_batch_size=300)


def test_nvidia_embedding_mode_switch_throws_without_key():
    emb = NVIDIAEmbedding()
    with pytest.raises(ValueError):
        emb.mode("nvidia")


def test_nvidia_embedding_mode_switch_throws_without_url():
    emb = NVIDIAEmbedding()
    with pytest.raises(ValueError):
        emb.mode("nim")


def test_nvidia_embedding_mode_switch_param_setting():
    emb = NVIDIAEmbedding()

    nim_emb = emb.mode("nim", base_url="https://test_url/v1/", model="dummy")
    assert nim_emb.model == "dummy"
    assert str(nim_emb._client.base_url) == "https://test_url/v1/"
    assert str(nim_emb._aclient.base_url) == "https://test_url/v1/"

    cat_emb = nim_emb.mode("nvidia", api_key="test", model="dummy-2")
    assert cat_emb.model == "dummy-2"
    assert (
        str(cat_emb._client.base_url)
        == "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    )
    assert (
        str(cat_emb._aclient.base_url)
        == "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    )
    assert cat_emb._client.api_key == "test"
    assert cat_emb._aclient.api_key == "test"


def test_nvidia_embedding_async():
    emb = NVIDIAEmbedding()

    assert inspect.iscoroutinefunction(emb._aget_query_embedding)
    query_emb = emb._aget_query_embedding("hi")
    assert inspect.isawaitable(query_emb)
    query_emb.close()

    assert inspect.iscoroutinefunction(emb._aget_text_embedding)
    text_emb = emb._aget_text_embedding("hi")
    assert inspect.isawaitable(text_emb)
    text_emb.close()

    assert inspect.iscoroutinefunction(emb._aget_text_embeddings)
    text_embs = emb._aget_text_embeddings(["hi", "hello"])
    assert inspect.isawaitable(text_embs)
    text_embs.close()


def test_nvidia_embedding_callback():
    llama_debug = LlamaDebugHandler(print_trace_on_end=False)
    assert len(llama_debug.get_events()) == 0

    callback_manager = CallbackManager([llama_debug])
    emb = NVIDIAEmbedding(api_key="dummy", callback_manager=callback_manager)

    try:
        emb.get_text_embedding("hi")
    except AuthenticationError:
        pass

    assert len(llama_debug.get_events(CBEventType.EMBEDDING)) > 0


def test_nvidia_embedding_throws_with_invalid_key():
    emb = NVIDIAEmbedding(api_key="invalid")
    with pytest.raises(AuthenticationError):
        emb.get_text_embedding("hi")
