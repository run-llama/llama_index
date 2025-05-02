import pytest
from llama_index.embeddings.modelscope.base import ModelScopeEmbedding


@pytest.fixture()
def modelscope_embedding():
    return ModelScopeEmbedding()


@pytest.fixture()
def query():
    return "吃完海鲜可以喝牛奶吗?"


@pytest.fixture()
def text():
    return [
        "不可以，早晨喝牛奶不科学",
        "吃了海鲜后是不能再喝牛奶的，因为牛奶中含得有维生素C，如果海鲜喝牛奶一起服用会对人体造成一定的伤害",
        "吃海鲜是不能同时喝牛奶吃水果，这个至少间隔6小时以上才可以。",
        "吃海鲜是不可以吃柠檬的因为其中的维生素C会和海鲜中的矿物质形成砷",
    ]


@pytest.mark.single()
def test_modelscope_query(modelscope_embedding, query):
    sentence_embedding = modelscope_embedding.get_query_embedding(query)
    assert sentence_embedding is not None
    assert len(sentence_embedding) > 0


@pytest.mark.single()
def test_modelscope_text(modelscope_embedding, query):
    sentence_embedding = modelscope_embedding.get_text_embedding(query)
    assert sentence_embedding is not None
    assert len(sentence_embedding) > 0


@pytest.mark.batch()
def test_modelscope_text_embedding_batch(modelscope_embedding, text):
    sentence_embedding = modelscope_embedding.get_text_embedding_batch(text)
    assert sentence_embedding is not None
    assert len(sentence_embedding) == len(text)
    assert len(sentence_embedding[0]) > 0


@pytest.mark.asyncio()
async def test_modelscope_async_query(modelscope_embedding, query):
    sentence_embedding = await modelscope_embedding.aget_query_embedding(query)
    assert sentence_embedding is not None
    assert len(sentence_embedding) > 0
