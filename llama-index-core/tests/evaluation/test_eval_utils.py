import warnings

from llama_index.core.evaluation.eval_utils import get_responses


def test_get_responses_does_not_emit_asyncio_module_deprecation() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*asyncio_module.*", category=DeprecationWarning
        )
        assert get_responses([], query_engine=None) == []
