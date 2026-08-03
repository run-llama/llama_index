# LlamaIndex Tools Integration: LLM Sandbox

Runs agent-authored code inside an isolated container rather than on the host,
using [llm-sandbox](https://github.com/vndee/llm-sandbox). Docker, Podman and
Kubernetes backends; seven languages.

The container is hardened by default: no network, capped memory and pids, every
Linux capability dropped except `DAC_OVERRIDE`, and `no-new-privileges` set.

## Installation

```bash
pip install llama-index-tools-llm-sandbox
```

Requires a container runtime — Docker by default.

## Usage

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.llm_sandbox import LLMSandboxToolSpec

tool_spec = LLMSandboxToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
    system_prompt="Solve problems by writing and running Python. Always print results.",
)

print(await agent.run("What is the standard deviation of the first 50 primes?"))
```

Another language, or another backend:

```python
LLMSandboxToolSpec(lang="ruby")
LLMSandboxToolSpec(backend="kubernetes")
LLMSandboxToolSpec(image="my-registry/python-with-pandas:1.0")
```

## Security

`DEFAULT_RUNTIME_CONFIGS` is applied unless you pass `runtime_configs`:

```python
{
    "network_mode": "none",
    "mem_limit": "512m",
    "pids_limit": 128,
    "cap_drop": ["ALL"],
    "cap_add": ["DAC_OVERRIDE"],
    "security_opt": ["no-new-privileges:true"],
}
```

Verified inside the container: `CapEff` is `0000000000000002` and outbound
connections fail.

Three things worth knowing:

- **`DAC_OVERRIDE` is added back deliberately.** The session copies the source
  file into the container; without that capability it cannot read it and every
  run fails. Everything else stays dropped.
- **`read_only: True` is not supported.** Docker rejects the code copy against a
  read-only rootfs, with or without a tmpfs on the workdir.
- **There is no package-installation parameter.** The default network isolation
  would block it, and letting a model choose arbitrary PyPI packages executes
  `setup.py` at install time. Pre-bake dependencies into an image and pass
  `image=` instead.

This is container isolation, not VM isolation: it inherits the threat model of
the backend in use and makes no kernel-level guarantee. For deliberately
adversarial code, pair it with a hardened runtime such as gVisor or Kata
Containers.
