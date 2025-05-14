# LlamaIndex LLM Integration: SambaNova LLM

SambaNova Systems LLMs are custom LLMs (Language Models) interfaces that allow you to interact with AI models hosted on SambaNova's offerings - SambaNova Cloud and SambaStudio

## Key Features:

- Integration with SambaNova-hosted AI models
- Integration two SambaNova offerings - SambaNova Cloud and SambaStudio
- Support for completion based interactions
- Streaming support for completion responses
- Seamless integration with the LlamaIndex ecosystem

## Installation

```bash
pip install llama-index-llms-sambanovasystems
```

## Usage

```python
from llama_index.llms.sambanovasystems import SambaNovaCloud

SambaNovaCloud(
    sambanova_url="SambaNova cloud endpoint URL",
    sambanova_api_key="set with your SambaNova cloud API key",
    model="model name",
    context_window=100000,
)
```

## Usage

```python
from llama_index.llms.sambanovasystems import SambaStudio

SambaStudio(
    sambastudio_url="SambaStudio endpoint URL",
    sambastudio_api_key="set with your SambaStudio endppoint API key",
    model="model name",
    context_window=100000,
)
```
