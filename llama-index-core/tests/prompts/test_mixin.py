"""Test prompt mixin."""

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)


class MockObject2(PromptMixin):
    def __init__(self) -> None:
        self._prompt_dict_2 = {
            "abc": PromptTemplate("{abc} {def}"),
        }

    def _get_prompts(self) -> PromptDictType:
        return self._prompt_dict_2

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "abc" in prompts:
            self._prompt_dict_2["abc"] = prompts["abc"]


class MockObject1(PromptMixin):
    def __init__(self) -> None:
        self.mock_object_2 = MockObject2()
        self._prompt_dict_1 = {
            "summary": PromptTemplate("{summary}"),
            "foo": PromptTemplate("{foo} {bar}"),
        }

    def _get_prompts(self) -> PromptDictType:
        return self._prompt_dict_1

    def _get_prompt_modules(self) -> PromptMixinType:
        return {"mock_object_2": self.mock_object_2}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "summary" in prompts:
            self._prompt_dict_1["summary"] = prompts["summary"]
        if "foo" in prompts:
            self._prompt_dict_1["foo"] = prompts["foo"]


def test_prompt_mixin() -> None:
    mock_obj1 = MockObject1()
    prompts = mock_obj1.get_prompts()
    assert prompts == {
        "summary": PromptTemplate("{summary}"),
        "foo": PromptTemplate("{foo} {bar}"),
        "mock_object_2:abc": PromptTemplate("{abc} {def}"),
    }

    assert mock_obj1.mock_object_2.get_prompts() == {
        "abc": PromptTemplate("{abc} {def}"),
    }

    # update prompts
    mock_obj1.update_prompts(
        {
            "summary": PromptTemplate("{summary} testing"),
            "mock_object_2:abc": PromptTemplate("{abc} {def} ghi"),
        }
    )
    assert mock_obj1.get_prompts() == {
        "summary": PromptTemplate("{summary} testing"),
        "foo": PromptTemplate("{foo} {bar}"),
        "mock_object_2:abc": PromptTemplate("{abc} {def} ghi"),
    }
