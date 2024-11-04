# LlamaIndex LLM Integration: SambaNova LLM

SambaNovaLLM is a custom LLM (Language Model) interface that allows you to interact with AI models hosted on SambaNova's offerings - SambaNova Cloud and SambaStudio

## Key Features:

- Integration with SambaNova-hosted AI models
- Integration two SambaNova offerings - SambaNova Cloud and SambaStudio
- Support for completion based interactions
- Streaming support for completion responses
- Seamless integration with the LlamaIndex ecosystem

## Installation

```bash
pip install llama-index-llms-sambanovacloud
```

## Usage

```python
from llama_index.llms.sambanovacloud import SambaNovaCloud

SambaNovaCloud(
    sambanova_url="SambaNova cloud endpoint URL",
    sambanova_api_key="set with your SambaNova cloud API key",
    model="model name",
)
```

## Usage

```python
SambaNovaCloud(
    sambanova_url="SambaNova cloud endpoint URL",
    sambanova_api_key="set with your SambaNova cloud API key",
    model="model name",
)
```
