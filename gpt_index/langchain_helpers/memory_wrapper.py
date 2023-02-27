"""Langchain memory wrapper (for LlamaIndex)."""

from typing import Any, Dict, List, Optional

from langchain.chains.base import Memory
from pydantic import Field

from gpt_index.indices.base import BaseGPTIndex
from gpt_index.readers.schema.base import Document


def get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    """Get prompt input key.

    Copied over from langchain.

    """
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]


class GPTIndexMemory(Memory):
    """Langchain memory wrapper (for LlamaIndex).

    Args:
        human_prefix (str): Prefix for human input. Defaults to "Human".
        ai_prefix (str): Prefix for AI output. Defaults to "AI".
        memory_key (str): Key for memory. Defaults to "history".
        index (BaseGPTIndex): LlamaIndex instance.
        query_kwargs (Dict[str, Any]): Keyword arguments for LlamaIndex query.
        input_key (Optional[str]): Input key. Defaults to None.
        output_key (Optional[str]): Output key. Defaults to None.

    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    index: BaseGPTIndex
    query_kwargs: Dict = Field(default_factory=dict)
    output_key: Optional[str] = None
    input_key: Optional[str] = None

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        return prompt_input_key

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        prompt_input_key = self._get_prompt_input_key(inputs)
        query_str = inputs[prompt_input_key]

        # TODO: wrap in prompt
        # TODO: add option to return the raw text
        # NOTE: currently it's a hack
        response = self.index.query(query_str, **self.query_kwargs)
        return {self.memory_key: str(response)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""
        prompt_input_key = self._get_prompt_input_key(inputs)
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        doc_text = "\n".join([human, ai])
        doc = Document(text=doc_text)
        self.index.insert(doc)

    def clear(self) -> None:
        """Clear memory contents."""
        pass

    def __repr__(self) -> str:
        """Return representation."""
        return "GPTIndexMemory()"
