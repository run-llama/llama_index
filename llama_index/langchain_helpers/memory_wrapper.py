"""Langchain memory wrapper (for LlamaIndex)."""

from typing import Any, Dict, List, Optional

from llama_index.bridge.langchain import (
    AIMessage,
    BaseChatMemory,
    BaseMessage,
    HumanMessage,
)
from llama_index.bridge.langchain import BaseMemory as Memory
from llama_index.bridge.pydantic import Field
from llama_index.indices.base import BaseIndex
from llama_index.schema import Document
from llama_index.utils import get_new_id


def get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    """Get prompt input key.

    Copied over from langchain.

    """
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference([*memory_variables, "stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]


class GPTIndexMemory(Memory):
    """Langchain memory wrapper (for LlamaIndex).

    Args:
        human_prefix (str): Prefix for human input. Defaults to "Human".
        ai_prefix (str): Prefix for AI output. Defaults to "AI".
        memory_key (str): Key for memory. Defaults to "history".
        index (BaseIndex): LlamaIndex instance.
        query_kwargs (Dict[str, Any]): Keyword arguments for LlamaIndex query.
        input_key (Optional[str]): Input key. Defaults to None.
        output_key (Optional[str]): Output key. Defaults to None.

    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    index: BaseIndex
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
        query_engine = self.index.as_query_engine(**self.query_kwargs)
        response = query_engine.query(query_str)
        return {self.memory_key: str(response)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""
        prompt_input_key = self._get_prompt_input_key(inputs)
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = next(iter(outputs.keys()))
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        doc_text = f"{human}\n{ai}"
        doc = Document(text=doc_text)
        self.index.insert(doc)

    def clear(self) -> None:
        """Clear memory contents."""

    def __repr__(self) -> str:
        """Return representation."""
        return "GPTIndexMemory()"


class GPTIndexChatMemory(BaseChatMemory):
    """Langchain chat memory wrapper (for LlamaIndex).

    Args:
        human_prefix (str): Prefix for human input. Defaults to "Human".
        ai_prefix (str): Prefix for AI output. Defaults to "AI".
        memory_key (str): Key for memory. Defaults to "history".
        index (BaseIndex): LlamaIndex instance.
        query_kwargs (Dict[str, Any]): Keyword arguments for LlamaIndex query.
        input_key (Optional[str]): Input key. Defaults to None.
        output_key (Optional[str]): Output key. Defaults to None.

    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    index: BaseIndex
    query_kwargs: Dict = Field(default_factory=dict)
    output_key: Optional[str] = None
    input_key: Optional[str] = None

    return_source: bool = False
    id_to_message: Dict[str, BaseMessage] = Field(default_factory=dict)

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

        query_engine = self.index.as_query_engine(**self.query_kwargs)
        response_obj = query_engine.query(query_str)
        if self.return_source:
            source_nodes = response_obj.source_nodes
            if self.return_messages:
                # get source messages from ids
                source_ids = [sn.node.node_id for sn in source_nodes]
                source_messages = [
                    m for id, m in self.id_to_message.items() if id in source_ids
                ]
                # NOTE: type List[BaseMessage]
                response: Any = source_messages
            else:
                source_texts = [sn.node.get_content() for sn in source_nodes]
                response = "\n\n".join(source_texts)
        else:
            response = str(response_obj)
        return {self.memory_key: response}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""
        prompt_input_key = self._get_prompt_input_key(inputs)
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = next(iter(outputs.keys()))
        else:
            output_key = self.output_key

        # a bit different than existing langchain implementation
        # because we want to track id's for messages
        human_message = HumanMessage(content=inputs[prompt_input_key])
        human_message_id = get_new_id(set(self.id_to_message.keys()))
        ai_message = AIMessage(content=outputs[output_key])
        ai_message_id = get_new_id(
            set(self.id_to_message.keys()).union({human_message_id})
        )

        self.chat_memory.messages.append(human_message)
        self.chat_memory.messages.append(ai_message)

        self.id_to_message[human_message_id] = human_message
        self.id_to_message[ai_message_id] = ai_message

        human_txt = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai_txt = f"{self.ai_prefix}: " + outputs[output_key]
        human_doc = Document(text=human_txt, id_=human_message_id)
        ai_doc = Document(text=ai_txt, id_=ai_message_id)
        self.index.insert(human_doc)
        self.index.insert(ai_doc)

    def clear(self) -> None:
        """Clear memory contents."""

    def __repr__(self) -> str:
        """Return representation."""
        return "GPTIndexMemory()"
