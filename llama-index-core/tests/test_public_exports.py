from importlib import import_module

import pytest


MODULES_WITH_PUBLIC_EXPORTS = (
    "llama_index.core",
    "llama_index.core.indices",
    "llama_index.core.prompts",
    "llama_index.core.indices.knowledge_graph",
)

REMOVED_EXPORTS = {
    "llama_index.core": {
        "GPTKnowledgeGraphIndex",
        "SummaryPrompt",
        "TreeInsertPrompt",
        "TreeSelectPrompt",
        "TreeSelectMultiplePrompt",
        "RefinePrompt",
        "QuestionAnswerPrompt",
        "KeywordExtractPrompt",
        "QueryKeywordExtractPrompt",
    },
    "llama_index.core.indices": {"GPTKnowledgeGraphIndex"},
}


@pytest.mark.parametrize("module_name", MODULES_WITH_PUBLIC_EXPORTS)
def test_public_exports_are_bound(module_name: str) -> None:
    module = import_module(module_name)

    missing_exports = [name for name in module.__all__ if not hasattr(module, name)]

    assert missing_exports == []


@pytest.mark.parametrize(
    ("module_name", "removed_exports"),
    REMOVED_EXPORTS.items(),
)
def test_removed_stale_exports_are_not_listed(
    module_name: str, removed_exports: set[str]
) -> None:
    module = import_module(module_name)

    assert removed_exports.isdisjoint(module.__all__)
