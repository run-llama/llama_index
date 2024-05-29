# Zenguard Guardrails LLamaPack

This LlamaPack shows how to use an implementation of Zenguard Guardrails with llama-index.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack ZenguardGuardrailsPack --download-dir ./zenguard_guardrails_pack
```

You can then inspect/modify the files at `./zenguard_guardrails_pack` and use them as a template for your own project.

## Code Usage

You can alternaitvely install the package:

`pip install llama-index-packs-zenguard-guardrails`

Then, you can import and initialize the pack!

```python
from llama_index.packs.zenguard_guardrails import ZenguardGuardrailsPack, ZenGuardConfig, Credentials

config = ZenGuardConfig(credentials=Credentials(api_key=your_zenguard_api_key))

pack = ZenguardGuardrailsPack(config)
```

The `run()` function is a light wrapper around `zenguard.detect()`.

```python
from llama_index.packs.zenguard_guardrails import Detector

response = pack.run(
    prompt = your_prompt,
    detectors = [Detector.PROMPT_INJECTION]
)
```

You can also use modules individually.

```python
# get the guardrails
guardrails = pack.get_modules()["guardrails"]
```