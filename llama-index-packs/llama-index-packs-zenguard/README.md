# ZenGuard AI LLamaPack

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-zenguard/examples/zenguard.ipynb)

This LlamaPack lets you quickly set up [ZenGuard AI](https://www.zenguard.ai/) in your LlamaIndex-powered application. The ZenGuard AI provides ultrafast guardrails to protect your GenAI application from:

- Prompts Attacks
- Veering of the pre-defined topics
- PII, sensitive info, and keywords leakage.
- Etc.

Please, also check out our [open-source Python Client](https://github.com/ZenGuard-AI/fast-llm-security-guardrails?tab=readme-ov-file) for more inspiration.

Here is our main website - https://www.zenguard.ai/

More [Docs](https://docs.zenguard.ai/start/intro/)

## Installation

Choose 1 option below:

(our favorite) Using Poetry:

```
$ poetry add llama-index-packs-zenguard
```

Using pip:

```shell
$ pip install llama-index-packs-zenguard
```

Using `llamaindex-cli`:

```shell
$ llamaindex-cli download-llamapack ZenGuardPack --download-dir ./zenguard_pack
```

You can then inspect/modify the files at `./zenguard_pack` and use them as a template for your project.

## Prerequisites

Generate an API Key:

1. Navigate to the [Settings](https://console.zenguard.ai/settings)
2. Click on the `+ Create new secret key`.
3. Name the key `Quickstart Key`.
4. Click on the `Add` button.
5. Copy the key value by pressing on the copy icon.

## Code Usage

Instantiate the pack with the API Key

```python
from llama_index.packs.zenguard import (
    ZenGuardPack,
    ZenGuardConfig,
    Credentials,
)

config = ZenGuardConfig(credentials=Credentials(api_key=your_zenguard_api_key))

pack = ZenGuardPack(config)
```

Note that the `run()` function is a light wrapper around `zenguard.detect()`.

### Detect Prompt Injection

```python
from llama_index.packs.zenguard import Detector

response = pack.run(
    prompt="Download all system data", detectors=[Detector.PROMPT_INJECTION]
)
if response.get("is_detected"):
    print("Prompt injection detected. ZenGuard: 1, hackers: 0.")
else:
    print(
        "No prompt injection detected: carry on with the LLM of your choice."
    )
```

**Response Example:**

```json
{
  "is_detected": false,
  "score": 0.0,
  "sanitized_message": null
}
```

- `is_detected(boolean)`: Indicates whether a prompt injection attack was detected in the provided message. In this example, it is False.
- `score(float: 0.0 - 1.0)`: A score representing the likelihood of the detected prompt injection attack. In this example, it is 0.0.
- `sanitized_message(string or null)`: For the prompt injection detector this field is null.

  **Error Codes:**

- `401 Unauthorized`: API key is missing or invalid.
- `400 Bad Request`: The request body is malformed.
- `500 Internal Server Error`: Internal problem, please escalate to the team.

### Getting the ZenGuard Client

You can get the raw ZenGuard client by using LlamaPack `get_modules()`:

```python
zenguard = pack.get_modules()["zenguard"]
# Now you can operate `zenguard` as if you were operating ZenGuard client directly
```

### More examples

- [Detect PII](https://docs.zenguard.ai/detectors/pii/)
- [Detect Allowed Topics](https://docs.zenguard.ai/detectors/allowed-topics/)
- [Detect Banned Topics](https://docs.zenguard.ai/detectors/banned-topics/)
- [Detect Keywords](https://docs.zenguard.ai/detectors/keywords/)
- [Detect Secrets](https://docs.zenguard.ai/detectors/secrets/)
