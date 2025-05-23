import logging
import os
from typing import Sequence

from llama_index.core.llms import LLM, MockLLM
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.base.base_selector import SelectorResult
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.selectors.llm_selectors import _build_choices_text

from notdiamond import NotDiamond, LLMConfig, Metric

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


class NotDiamondSelectorResult(SelectorResult):
    """A single selection of a choice provided by Not Diamond."""

    class Config:
        arbitrary_types_allowed = True

    session_id: str
    llm: LLMConfig

    @classmethod
    def from_selector_result(
        cls, selector_result: SelectorResult, session_id: str, best_llm: LLMConfig
    ) -> "NotDiamondSelectorResult":
        return cls(session_id=session_id, llm=best_llm, **selector_result.dict())


class NotDiamondSelector(LLMSingleSelector):
    def __init__(
        self,
        client: NotDiamond,
        metric: Metric = None,
        timeout: int = 10,
        api_key: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a NotDiamondSelector. Users should instantiate and configure a NotDiamond client as needed before
        creating this selector. The constructor will raise errors re: required client fields.
        """
        # Not needed - we will route using our own client based on the query prompt
        # Add @property for _llm here
        _encap_selector = LLMSingleSelector.from_defaults(llm=MockLLM())
        self._llm = None
        self._prompt = _encap_selector._prompt

        if not getattr(client, "llm_configs", None):
            raise ValueError(
                "NotDiamond client must have llm_configs before creating a NotDiamondSelector."
            )

        if metric and not isinstance(metric, Metric):
            raise ValueError(f"Invalid metric - needed type Metric but got {metric}")
        self._metric = metric or Metric("accuracy")

        self._client = client
        self._llms = [
            self._llm_config_to_client(llm_config)
            for llm_config in self._client.llm_configs
        ]
        self._timeout = timeout
        super().__init__(_encap_selector._llm, _encap_selector._prompt, *args, **kwargs)

    def _llm_config_to_client(self, llm_config: LLMConfig | str) -> LLM:
        """
        For the selected LLMConfig dynamically create an LLM instance. NotDiamondSelector will
        assign this to self._llm to help select the best index.
        """
        if isinstance(llm_config, str):
            llm_config = LLMConfig.from_string(llm_config)
        provider, model = llm_config.provider, llm_config.model

        output = None
        if provider == "openai":
            from llama_index.llms.openai import OpenAI

            output = OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic

            output = Anthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "cohere":
            from llama_index.llms.cohere import Cohere

            output = Cohere(model=model, api_key=os.getenv("COHERE_API_KEY"))
        elif provider == "mistral":
            from llama_index.llms.mistralai import MistralAI

            output = MistralAI(model=model, api_key=os.getenv("MISTRALAI_API_KEY"))
        elif provider == "togetherai":
            from llama_index.llms.together import TogetherLLM

            output = TogetherLLM(model=model, api_key=os.getenv("TOGETHERAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider for NotDiamondSelector: {provider}")

        return output

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle, timeout: int = None
    ) -> SelectorResult:
        """
        Call Not Diamond to select the best LLM for the given prompt, then have the LLM select the best tool.
        """
        messages = [
            {"role": "system", "content": self._format_prompt(choices, query)},
            {"role": "user", "content": query.query_str},
        ]

        session_id, best_llm = self._client.model_select(
            messages=messages,
            llm_configs=self._client.llm_configs,
            metric=self._metric,
            notdiamond_api_key=self._client.api_key,
            max_model_depth=self._client.max_model_depth,
            hash_content=self._client.hash_content,
            tradeoff=self._client.tradeoff,
            preference_id=self._client.preference_id,
            tools=self._client.tools,
            timeout=timeout or self._timeout,
        )

        self._llm = self._llm_config_to_client(best_llm)

        return NotDiamondSelectorResult.from_selector_result(
            super()._select(choices, query), session_id, best_llm
        )

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle, timeout: int = None
    ) -> SelectorResult:
        """
        Call Not Diamond asynchronously to select the best LLM for the given prompt, then have the LLM select the best tool.
        """
        messages = [
            {"role": "system", "content": self._format_prompt(choices, query)},
            {"role": "user", "content": query.query_str},
        ]

        session_id, best_llm = await self._client.amodel_select(
            messages=messages,
            llm_configs=self._client.llm_configs,
            metric=self._metric,
            notdiamond_api_key=self._client.api_key,
            max_model_depth=self._client.max_model_depth,
            hash_content=self._client.hash_content,
            tradeoff=self._client.tradeoff,
            preference_id=self._client.preference_id,
            tools=self._client.tools,
            timeout=timeout or self._timeout,
        )

        self._llm = self._llm_config_to_client(best_llm)

        return NotDiamondSelectorResult.from_selector_result(
            await super()._aselect(choices, query), session_id, best_llm
        )

    def _format_prompt(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> str:
        """
        A system prompt for selection is created when instantiating the parent LLMSingleSelector class.
        This method formats the prompt into a str so that it can be serialized for the NotDiamond API.
        """
        context_list = _build_choices_text(choices)
        return self._prompt.format(
            num_choices=len(choices),
            context_list=context_list,
            query_str=query.query_str,
        )
