from importlib import import_module


LEGACY_PROMPT_EXPORTS = [
    "SummaryPrompt",
    "TreeInsertPrompt",
    "TreeSelectPrompt",
    "TreeSelectMultiplePrompt",
    "RefinePrompt",
    "QuestionAnswerPrompt",
    "KeywordExtractPrompt",
    "QueryKeywordExtractPrompt",
]


def test_core_legacy_exports_are_bound() -> None:
    core = import_module("llama_index.core")

    assert core.GPTKnowledgeGraphIndex is core.KnowledgeGraphIndex
    for export_name in LEGACY_PROMPT_EXPORTS:
        assert getattr(core, export_name) is core.PromptTemplate


def test_prompts_legacy_exports_are_bound() -> None:
    prompts = import_module("llama_index.core.prompts")

    for export_name in LEGACY_PROMPT_EXPORTS:
        assert getattr(prompts, export_name) is prompts.PromptTemplate


def test_core_import_star_includes_legacy_exports() -> None:
    namespace: dict[str, object] = {}

    exec("from llama_index.core import *", namespace)

    assert namespace["GPTKnowledgeGraphIndex"] is namespace["KnowledgeGraphIndex"]
    for export_name in LEGACY_PROMPT_EXPORTS:
        assert namespace[export_name] is namespace["PromptTemplate"]
