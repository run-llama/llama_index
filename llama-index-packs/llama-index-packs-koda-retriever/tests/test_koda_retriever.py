from asyncio import run
from llama_index.packs.koda_retriever import KodaRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.llm import BaseLLM


def test_init(setup):
    retriever = setup.get("retriever")

    assert isinstance(retriever.index, BaseIndex), (
        "index should be an instance of BaseIndex"
    )
    assert isinstance(retriever.llm, BaseLLM), "llm should be an instance of LLM"
    assert isinstance(retriever, KodaRetriever), (
        "retriever should be an instance of KodaRetriever"
    )


def test_retrieve(setup):
    retriever = setup.get("retriever")
    llm = setup.get("llm")
    query = llm.random_prompt()
    results = retriever.retrieve(query)

    assert isinstance(results, list), "retrieve should return a list"


def test_a_retrieve(setup):
    retriever = setup.get("retriever")
    llm = setup.get("llm")
    query = llm.random_prompt()
    results = run(retriever.aretrieve(query))

    assert isinstance(results, list), "aretrieve should return a list"


def test_categorize(setup):
    retriever = setup.get("retriever")
    expected_categories = setup.get("matrix").get_categories()

    llm = setup.get("llm")
    query = llm.random_prompt()
    category = retriever.categorize(query)

    assert isinstance(category, str), "categorize should return a string"
    assert category in expected_categories, (
        "categorize should return a category from the matrix"
    )


def test_category_retrieve(setup):
    retriever = setup.get("retriever")
    llm = setup.get("llm")
    query = llm.random_prompt()
    category = llm.prompt_responses.get(query, "concept seeking query")

    results = retriever.category_retrieve(category, query)

    assert isinstance(results, list), "category_retrieve should return a list"
