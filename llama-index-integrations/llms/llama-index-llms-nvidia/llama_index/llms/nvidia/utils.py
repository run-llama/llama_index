from typing import Optional, Literal
import warnings

# from dataclasses import dataclass
from llama_index.core.bridge.pydantic import BaseModel, model_validator

DEFAULT_MODEL = "meta/llama3-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1/"


class Model(BaseModel):
    """
    Model information.
    id: unique identifier for the model, passed as model parameter for requests
    model_type: API type (chat, vlm, embedding, ranking, completions)
    client: client name, e.g. NvidiaGenerator, NVIDIAEmbeddings,
            NVIDIARerank, NvidiaTextEmbedder, NvidiaDocumentEmbedder
    endpoint: custom endpoint for the model
    aliases: list of aliases for the model
    supports_tools: whether the model supports tool calling
    supports_structured_output: whether the model supports structured output
    supports_thinking: whether the model supports thinking mode
    All aliases are deprecated and will trigger a warning when used.
    """

    id: str
    model_type: Literal["chat", "vlm", "completions"] = "chat"
    client: str = "NVIDIA"
    endpoint: Optional[str] = None
    aliases: Optional[list] = None
    base_model: Optional[str] = None
    supports_tools: Optional[bool] = False
    supports_structured_output: Optional[bool] = False
    supports_thinking: Optional[bool] = False

    def __hash__(self) -> int:
        return hash(self.id)

    @model_validator(mode="after")
    def validate_client(self) -> "Model":
        if self.client:
            supported = {"NVIDIA": ("chat", "vlm", "completions")}
            if self.model_type not in supported.get(self.client, ()):
                raise ValueError(
                    f"Model type '{self.model_type}' not supported "
                    f"by client '{self.client}'"
                )
        return self


CHAT_MODEL_TABLE = {
    "meta/codellama-70b": Model(
        id="meta/codellama-70b",
        model_type="chat",
        client="NVIDIA",
        aliases=[
            "ai-codellama-70b",
            "playground_llama2_code_70b",
            "llama2_code_70b",
            "playground_llama2_code_34b",
            "llama2_code_34b",
            "playground_llama2_code_13b",
            "llama2_code_13b",
        ],
    ),
    "google/gemma-7b": Model(
        id="google/gemma-7b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-gemma-7b", "playground_gemma_7b", "gemma_7b"],
    ),
    "meta/llama2-70b": Model(
        id="meta/llama2-70b",
        model_type="chat",
        client="NVIDIA",
        aliases=[
            "ai-llama2-70b",
            "playground_llama2_70b",
            "llama2_70b",
            "playground_llama2_13b",
            "llama2_13b",
        ],
    ),
    "mistralai/mistral-7b-instruct-v0.2": Model(
        id="mistralai/mistral-7b-instruct-v0.2",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-mistral-7b-instruct-v2", "playground_mistral_7b", "mistral_7b"],
    ),
    "mistralai/mixtral-8x7b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x7b-instruct-v0.1",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-mixtral-8x7b-instruct", "playground_mixtral_8x7b", "mixtral_8x7b"],
    ),
    "mistralai/mixtral-8x22b-v0.1": Model(
        id="mistralai/mixtral-8x22b-v0.1",
        model_type="chat",
        client="NVIDIA",
    ),
    "google/codegemma-7b": Model(
        id="google/codegemma-7b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-codegemma-7b"],
    ),
    "google/gemma-2b": Model(
        id="google/gemma-2b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-gemma-2b", "playground_gemma_2b", "gemma_2b"],
    ),
    "google/recurrentgemma-2b": Model(
        id="google/recurrentgemma-2b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-recurrentgemma-2b"],
    ),
    "gotocompany/gemma-2-9b-cpt-sahabatai-instruct": Model(
        id="gotocompany/gemma-2-9b-cpt-sahabatai-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "mistralai/mistral-large": Model(
        id="mistralai/mistral-large",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-mistral-large"],
    ),
    "mistralai/mixtral-8x22b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x22b-instruct-v0.1",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-mixtral-8x22b-instruct"],
    ),
    "meta/llama3-8b-instruct": Model(
        id="meta/llama3-8b-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-llama3-8b"],
    ),
    "meta/llama3-70b-instruct": Model(
        id="meta/llama3-70b-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-llama3-70b"],
    ),
    "microsoft/phi-3-mini-128k-instruct": Model(
        id="microsoft/phi-3-mini-128k-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-phi-3-mini"],
    ),
    "snowflake/arctic": Model(
        id="snowflake/arctic",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-arctic"],
    ),
    "databricks/dbrx-instruct": Model(
        id="databricks/dbrx-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-dbrx-instruct"],
    ),
    "microsoft/phi-3-mini-4k-instruct": Model(
        id="microsoft/phi-3-mini-4k-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-phi-3-mini-4k", "playground_phi2", "phi2"],
    ),
    "seallms/seallm-7b-v2.5": Model(
        id="seallms/seallm-7b-v2.5",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-seallm-7b"],
    ),
    "aisingapore/sea-lion-7b-instruct": Model(
        id="aisingapore/sea-lion-7b-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-sea-lion-7b-instruct"],
    ),
    "microsoft/phi-3-small-8k-instruct": Model(
        id="microsoft/phi-3-small-8k-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-phi-3-small-8k-instruct"],
    ),
    "microsoft/phi-3-small-128k-instruct": Model(
        id="microsoft/phi-3-small-128k-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-phi-3-small-128k-instruct"],
    ),
    "microsoft/phi-3-medium-4k-instruct": Model(
        id="microsoft/phi-3-medium-4k-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-phi-3-medium-4k-instruct"],
    ),
    "ibm/granite-8b-code-instruct": Model(
        id="ibm/granite-8b-code-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-granite-8b-code-instruct"],
    ),
    "ibm/granite-34b-code-instruct": Model(
        id="ibm/granite-34b-code-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-granite-34b-code-instruct"],
    ),
    "ibm/granite-3.3-8b-instruct": Model(
        id="ibm/granite-3.3-8b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
    "ibm/granite-guardian-3.0-8b": Model(
        id="ibm/granite-guardian-3.0-8b",
        model_type="chat",
        client="NVIDIA",
    ),
    "google/codegemma-1.1-7b": Model(
        id="google/codegemma-1.1-7b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-codegemma-1.1-7b"],
    ),
    "mediatek/breeze-7b-instruct": Model(
        id="mediatek/breeze-7b-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-breeze-7b-instruct"],
    ),
    "upstage/solar-10.7b-instruct": Model(
        id="upstage/solar-10.7b-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-solar-10_7b-instruct"],
    ),
    "writer/palmyra-med-70b-32k": Model(
        id="writer/palmyra-med-70b-32k",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-palmyra-med-70b-32k"],
    ),
    "writer/palmyra-med-70b": Model(
        id="writer/palmyra-med-70b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-palmyra-med-70b"],
    ),
    "mistralai/mistral-7b-instruct-v0.3": Model(
        id="mistralai/mistral-7b-instruct-v0.3",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-mistral-7b-instruct-v03"],
    ),
    "01-ai/yi-large": Model(
        id="01-ai/yi-large",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-yi-large"],
    ),
    "nvidia/nemotron-4-340b-instruct": Model(
        id="nvidia/nemotron-4-340b-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["qa-nemotron-4-340b-instruct"],
    ),
    "nvidia/nemotron-4-340b-reward": Model(
        id="nvidia/nemotron-4-340b-reward",
        model_type="chat",
        client="NVIDIA",
    ),
    "mistralai/codestral-22b-instruct-v0.1": Model(
        id="mistralai/codestral-22b-instruct-v0.1",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-codestral-22b-instruct-v01"],
        supports_structured_output=True,
    ),
    "google/gemma-2-9b-it": Model(
        id="google/gemma-2-9b-it",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-gemma-2-9b-it"],
    ),
    "google/gemma-2-27b-it": Model(
        id="google/gemma-2-27b-it",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-gemma-2-27b-it"],
    ),
    "google/gemma-3-12b-it": Model(
        id="google/gemma-3-12b-it",
        model_type="chat",
        client="NVIDIA",
    ),
    "google/gemma-3-1b-it": Model(
        id="google/gemma-3-1b-it",
        model_type="chat",
        client="NVIDIA",
    ),
    "google/gemma-3-27b-it": Model(
        id="google/gemma-3-27b-it",
        model_type="chat",
        client="NVIDIA",
    ),
    "google/gemma-3-4b-it": Model(
        id="google/gemma-3-4b-it",
        model_type="chat",
        client="NVIDIA",
    ),
    "google/shieldgemma-9b": Model(
        id="google/shieldgemma-9b",
        model_type="chat",
        client="NVIDIA",
    ),
    "microsoft/phi-3-medium-128k-instruct": Model(
        id="microsoft/phi-3-medium-128k-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-phi-3-medium-128k-instruct"],
    ),
    "deepseek-ai/deepseek-coder-6.7b-instruct": Model(
        id="deepseek-ai/deepseek-coder-6.7b-instruct",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-deepseek-coder-6_7b-instruct"],
    ),
    "nv-mistralai/mistral-nemo-12b-instruct": Model(
        id="nv-mistralai/mistral-nemo-12b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "meta/llama-3.1-8b-instruct": Model(
        id="meta/llama-3.1-8b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "meta/llama-3.1-70b-instruct": Model(
        id="meta/llama-3.1-70b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "meta/llama-3.1-405b-instruct": Model(
        id="meta/llama-3.1-405b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "meta/llama-guard-4-12b": Model(
        id="meta/llama-guard-4-12b",
        model_type="chat",
        client="NVIDIA",
    ),
    "nvidia/usdcode-llama3-70b-instruct": Model(
        id="nvidia/usdcode-llama3-70b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "mistralai/mamba-codestral-7b-v0.1": Model(
        id="mistralai/mamba-codestral-7b-v0.1",
        model_type="chat",
        client="NVIDIA",
    ),
    "writer/palmyra-fin-70b-32k": Model(
        id="writer/palmyra-fin-70b-32k",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "google/gemma-2-2b-it": Model(
        id="google/gemma-2-2b-it",
        model_type="chat",
        client="NVIDIA",
    ),
    "mistralai/mistral-large-2-instruct": Model(
        id="mistralai/mistral-large-2-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "mistralai/magistral-small-2506": Model(
        id="mistralai/magistral-small-2506",
        model_type="chat",
        client="NVIDIA",
    ),
    "mistralai/mistral-nemotron": Model(
        id="mistralai/mistral-nemotron",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "mistralai/mistral-small-24b-instruct": Model(
        id="mistralai/mistral-small-24b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "mistralai/mathstral-7b-v0.1": Model(
        id="mistralai/mathstral-7b-v0.1",
        model_type="chat",
        client="NVIDIA",
    ),
    "rakuten/rakutenai-7b-instruct": Model(
        id="rakuten/rakutenai-7b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "rakuten/rakutenai-7b-chat": Model(
        id="rakuten/rakutenai-7b-chat",
        model_type="chat",
        client="NVIDIA",
    ),
    "baichuan-inc/baichuan2-13b-chat": Model(
        id="baichuan-inc/baichuan2-13b-chat",
        model_type="chat",
        client="NVIDIA",
    ),
    "thudm/chatglm3-6b": Model(
        id="thudm/chatglm3-6b",
        model_type="chat",
        client="NVIDIA",
    ),
    "tiiuae/falcon3-7b-instruct": Model(
        id="tiiuae/falcon3-7b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "microsoft/phi-3.5-mini-instruct": Model(
        id="microsoft/phi-3.5-mini-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "microsoft/phi-3.5-moe-instruct": Model(
        id="microsoft/phi-3.5-moe-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "microsoft/phi-4-mini-flash-reasoning": Model(
        id="microsoft/phi-4-mini-flash-reasoning",
        model_type="chat",
        client="NVIDIA",
    ),
    "microsoft/phi-4-mini-instruct": Model(
        id="microsoft/phi-4-mini-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "utter-project/eurollm-9b-instruct": Model(
        id="utter-project/eurollm-9b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "writer/palmyra-creative-122b": Model(
        id="writer/palmyra-creative-122b",
        model_type="chat",
        client="NVIDIA",
    ),
    "nvidia/nemotron-mini-4b-instruct": Model(
        id="nvidia/nemotron-mini-4b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "ai21labs/jamba-1.5-large-instruct": Model(
        id="ai21labs/jamba-1.5-large-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "ai21labs/jamba-1.5-mini-instruct": Model(
        id="ai21labs/jamba-1.5-mini-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "yentinglin/llama-3-taiwan-70b-instruct": Model(
        id="yentinglin/llama-3-taiwan-70b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "tokyotech-llm/llama-3-swallow-70b-instruct-v0.1": Model(
        id="tokyotech-llm/llama-3-swallow-70b-instruct-v0.1",
        model_type="chat",
        client="NVIDIA",
    ),
    "abacusai/dracarys-llama-3.1-70b-instruct": Model(
        id="abacusai/dracarys-llama-3.1-70b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "qwen/qwen2-7b-instruct": Model(
        id="qwen/qwen2-7b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "nvidia/llama-3.1-nemotron-51b-instruct": Model(
        id="nvidia/llama-3.1-nemotron-51b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "meta/llama-3.2-1b-instruct": Model(
        id="meta/llama-3.2-1b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "meta/llama-3.2-3b-instruct": Model(
        id="meta/llama-3.2-3b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "nvidia/mistral-nemo-minitron-8b-8k-instruct": Model(
        id="nvidia/mistral-nemo-minitron-8b-8k-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "nvidia/nemoretriever-parse": Model(
        id="nvidia/nemoretriever-parse",
        model_type="chat",
        client="NVIDIA",
    ),
    "nvidia/riva-translate-4b-instruct": Model(
        id="nvidia/riva-translate-4b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "institute-of-science-tokyo/llama-3.1-swallow-8b-instruct-v0.1": Model(
        id="institute-of-science-tokyo/llama-3.1-swallow-8b-instruct-v0.1",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "institute-of-science-tokyo/llama-3.1-swallow-70b-instruct-v0.1": Model(
        id="institute-of-science-tokyo/llama-3.1-swallow-70b-instruct-v0.1",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "igenius/colosseum_355b_instruct_16k": Model(
        id="igenius/colosseum_355b_instruct_16k",
        model_type="chat",
        client="NVIDIA",
    ),
    "igenius/italia_10b_instruct_16k": Model(
        id="igenius/italia_10b_instruct_16k",
        model_type="chat",
        client="NVIDIA",
    ),
    "zyphra/zamba2-7b-instruct": Model(
        id="zyphra/zamba2-7b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "ibm/granite-3.0-8b-instruct": Model(
        id="ibm/granite-3.0-8b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "ibm/granite-3.0-3b-a800m-instruct": Model(
        id="ibm/granite-3.0-3b-a800m-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "nvidia/nemotron-4-mini-hindi-4b-instruct": Model(
        id="nvidia/nemotron-4-mini-hindi-4b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "nvidia/llama-3.1-nemotron-70b-instruct": Model(
        id="nvidia/llama-3.1-nemotron-70b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "nvidia/llama-3.1-nemoguard-8b-content-safety": Model(
        id="nvidia/llama-3.1-nemoguard-8b-content-safety",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "nvidia/llama-3.1-nemoguard-8b-topic-control": Model(
        id="nvidia/llama-3.1-nemoguard-8b-topic-control",
        model_type="chat",
        client="NVIDIA",
    ),
    "nvidia/usdcode-llama-3.1-70b-instruct": Model(
        id="nvidia/usdcode-llama-3.1-70b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "meta/llama-3.3-70b-instruct": Model(
        id="meta/llama-3.3-70b-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "nvidia/llama3-chatqa-1.5-8b": Model(
        id="nvidia/llama3-chatqa-1.5-8b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-chatqa-1.5-8b"],
    ),
    "nvidia/llama3-chatqa-1.5-70b": Model(
        id="nvidia/llama3-chatqa-1.5-70b",
        model_type="chat",
        client="NVIDIA",
        aliases=["ai-chatqa-1.5-70b"],
    ),
    "qwen/qwen2.5-coder-32b-instruct": Model(
        id="qwen/qwen2.5-coder-32b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "qwen/qwen2.5-coder-7b-instruct": Model(
        id="qwen/qwen2.5-coder-7b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "qwen/qwen2.5-7b-instruct": Model(
        id="qwen/qwen2.5-7b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "qwen/qwen3-235b-a22b": Model(
        id="qwen/qwen3-235b-a22b",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
    "qwen/qwq-32b": Model(
        id="qwen/qwq-32b",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
    "nvidia/llama-3.1-nemotron-70b-reward": Model(
        id="nvidia/llama-3.1-nemotron-70b-reward",
        model_type="chat",
        client="NVIDIA",
    ),
    "deepseek-ai/deepseek-r1": Model(
        id="deepseek-ai/deepseek-r1",
        model_type="chat",
        client="NVIDIA",
    ),
    "deepseek-ai/deepseek-r1-0528": Model(
        id="deepseek-ai/deepseek-r1-0528",
        model_type="chat",
        client="NVIDIA",
        supports_structured_output=True,
    ),
    "deepseek-ai/deepseek-r1-distill-llama-8b": Model(
        id="deepseek-ai/deepseek-r1-distill-llama-8b",
        model_type="chat",
        client="NVIDIA",
    ),
    "deepseek-ai/deepseek-r1-distill-qwen-14b": Model(
        id="deepseek-ai/deepseek-r1-distill-qwen-14b",
        model_type="chat",
        client="NVIDIA",
    ),
    "deepseek-ai/deepseek-r1-distill-qwen-32b": Model(
        id="deepseek-ai/deepseek-r1-distill-qwen-32b",
        model_type="chat",
        client="NVIDIA",
    ),
    "deepseek-ai/deepseek-r1-distill-qwen-7b": Model(
        id="deepseek-ai/deepseek-r1-distill-qwen-7b",
        model_type="chat",
        client="NVIDIA",
    ),
    "nvidia/llama-3.1-nemotron-nano-8b-v1": Model(
        id="nvidia/llama-3.1-nemotron-nano-8b-v1",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
    ),
    "nvidia/llama-3.1-nemotron-nano-4b-v1.1": Model(
        id="nvidia/llama-3.1-nemotron-nano-4b-v1.1",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": Model(
        id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
    "nvidia/llama-3.3-nemotron-super-49b-v1": Model(
        id="nvidia/llama-3.3-nemotron-super-49b-v1",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": Model(
        id="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        model_type="chat",
        client="NVIDIA",
        supports_thinking=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
    "moonshotai/kimi-k2-instruct": Model(
        id="moonshotai/kimi-k2-instruct",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "openai/gpt-oss-20b": Model(
        id="openai/gpt-oss-20b",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "openai/gpt-oss-120b": Model(
        id="openai/gpt-oss-120b",
        model_type="chat",
        client="NVIDIA",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "marin/marin-8b-instruct": Model(
        id="marin/marin-8b-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
    "speakleash/bielik-11b-v2.3-instruct": Model(
        id="speakleash/bielik-11b-v2.3-instruct",
        model_type="chat",
        client="NVIDIA",
    ),
}

VLM_MODEL_TABLE = {
    "adept/fuyu-8b": Model(
        id="adept/fuyu-8b",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b",
        aliases=["ai-fuyu-8b", "playground_fuyu_8b", "fuyu_8b"],
    ),
    "google/deplot": Model(
        id="google/deplot",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/google/deplot",
        aliases=["ai-google-deplot", "playground_deplot", "deplot"],
    ),
    "microsoft/kosmos-2": Model(
        id="microsoft/kosmos-2",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2",
        aliases=["ai-microsoft-kosmos-2", "playground_kosmos_2", "kosmos_2"],
    ),
    "nvidia/neva-22b": Model(
        id="nvidia/neva-22b",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b",
        aliases=["ai-neva-22b", "playground_neva_22b", "neva_22b"],
    ),
    "google/paligemma": Model(
        id="google/paligemma",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/google/paligemma",
        aliases=["ai-google-paligemma"],
    ),
    "microsoft/phi-3-vision-128k-instruct": Model(
        id="microsoft/phi-3-vision-128k-instruct",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct",
        aliases=["ai-phi-3-vision-128k-instruct"],
    ),
    "microsoft/phi-3.5-vision-instruct": Model(
        id="microsoft/phi-3.5-vision-instruct",
        model_type="vlm",
        client="NVIDIA",
    ),
    "microsoft/phi-4-multimodal-instruct": Model(
        id="microsoft/phi-4-multimodal-instruct",
        model_type="vlm",
        client="NVIDIA",
    ),
    "nvidia/vila": Model(
        id="nvidia/vila",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/nvidia/vila",
    ),
    "nvidia/nvclip": Model(
        id="nvidia/nvclip",
        model_type="vlm",
        client="NVIDIA",
    ),
    "meta/llama-3.2-11b-vision-instruct": Model(
        id="meta/llama-3.2-11b-vision-instruct",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions",
    ),
    "meta/llama-3.2-90b-vision-instruct": Model(
        id="meta/llama-3.2-90b-vision-instruct",
        model_type="vlm",
        client="NVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions",
    ),
    "meta/llama-4-maverick-17b-128e-instruct": Model(
        id="meta/llama-4-maverick-17b-128e-instruct",
        model_type="vlm",
        client="NVIDIA",
    ),
    "meta/llama-4-scout-17b-16e-instruct": Model(
        id="meta/llama-4-scout-17b-16e-instruct",
        model_type="vlm",
        client="NVIDIA",
    ),
    "google/gemma-3n-e2b-it": Model(
        id="google/gemma-3n-e2b-it",
        model_type="vlm",
        client="NVIDIA",
    ),
    "google/gemma-3n-e4b-it": Model(
        id="google/gemma-3n-e4b-it",
        model_type="vlm",
        client="NVIDIA",
    ),
    "mistralai/mistral-medium-3-instruct": Model(
        id="mistralai/mistral-medium-3-instruct",
        model_type="vlm",
        client="NVIDIA",
    ),
    "mistralai/mistral-small-3.1-24b-instruct-2503": Model(
        id="mistralai/mistral-small-3.1-24b-instruct-2503",
        model_type="vlm",
        client="NVIDIA",
    ),
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": Model(
        id="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        model_type="vlm",
        client="NVIDIA",
    ),
}

COMPLETION_MODEL_TABLE = {
    "bigcode/starcoder2-7b": Model(
        id="bigcode/starcoder2-7b",
        model_type="completions",
        client="NVIDIA",
    ),
    "bigcode/starcoder2-15b": Model(
        id="bigcode/starcoder2-15b",
        model_type="completions",
        client="NVIDIA",
    ),
    "nvidia/mistral-nemo-minitron-8b-base": Model(
        id="nvidia/mistral-nemo-minitron-8b-base",
        model_type="completions",
        client="NVIDIA",
    ),
}

MODEL_TABLE = {**CHAT_MODEL_TABLE, **VLM_MODEL_TABLE, **COMPLETION_MODEL_TABLE}


def lookup_model(name: str) -> Optional[Model]:
    """
    Lookup a model by name, using only the table of known models.
    The name is either:
        - directly in the table
        - an alias in the table
        - not found (None)
    Callers can check to see if the name was an alias by
    comparing the result's id field to the name they provided.
    """
    model = None
    if not (model := MODEL_TABLE.get(name)):
        for mdl in MODEL_TABLE.values():
            if mdl.aliases and name in mdl.aliases:
                model = mdl
                break
    return model


def determine_model(name: str) -> Optional[Model]:
    """
    Determine the model to use based on a name, using
    only the table of known models.

    Raise a warning if the model is found to be
    an alias of a known model.

    If the model is not found, return None.
    """
    if model := lookup_model(name):
        # all aliases are deprecated
        if model.id != name:
            warnings.warn(
                f"Model {name} is deprecated. Using {model.id} instead.", UserWarning
            )
    return model
