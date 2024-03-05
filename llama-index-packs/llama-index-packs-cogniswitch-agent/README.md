## Cogniswitch LlamaPack

**Use CogniSwitch to build production ready applications that can consume, organize and retrieve knowledge flawlessly. Using the framework of your choice, in this case LlamaIndex, CogniSwitch helps alleviate the stress of decision making when it comes to, choosing the right storage and retrieval formats. It also eradicates reliability issues and hallucinations when it comes to responses that are generated. Get started by interacting with your knowledge in a few simple steps**

visit [https://www.cogniswitch.ai/developer](https://www.cogniswitch.ai/developer?utm_source=llamaindex&utm_medium=llamaindexbuild&utm_id=dev).

**Registration:**

- Signup with your email and verify your registration
- You will get a mail with a platform token and OAuth token for using the services.

**Step 1: Download the CogniSwitch Llama pack:**

- Download the CogniswitchAgentPack either with the llama-cli or import using the code.

**Step 2: Instantiate the CogniswitchAgentPack:**

- Instantiate the cogniswitch agent pack with all the credentials.

**Step 3: Cogniswitch Store data:**

- Make the call to the agent by giving the file path or url to the agent input.
- The agent will pick the tool and use the file/url and it will be processed and stored in your knowledge store.
- You can check the status of document processing with a call to the agent. Alternatively you can also check in [cogniswitch console](https://console.cogniswitch.ai:8443/login?utm_source=llamaindex&utm_medium=llamaindexbuild&utm_id=dev).

**Step 4: Cogniswitch Answer:**

- Make the call to the agent by giving query as agent input.
- You will get the answer from your knowledge as the response.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack CogniswitchAgentPack --download-dir ./cs_pack
```

## Code Usage

```python
import warnings

warnings.filterwarnings("ignore")
from llama_index.packs.cogniswitch_agent import CogniswitchAgentPack
import os


### Cogniswitch Credentials and OpenAI token
# os.environ["OPENAI_API_KEY"] = <your openai token>
# cogniswitch_tool_args = {
#   "cs_token":<your cogniswitch platform token>,
#   "apiKey":<your cogniswitch apikey>
# }

cogniswitch_agent_pack = CogniswitchAgentPack(cogniswitch_tool_args)
```

From here, you can use the pack, or inspect and modify the pack in `./cs_pack`.

The `run()` function is a light wrapper around `agent.chat()`.

### Use the cogniswitch agent for storing data in cogniswitch with a single call

```python
response = cogniswitch_agent_pack.run(
    "Upload this URL- https://cogniswitch.ai/developer"
)
```

### Use the cogniswitch agent to know the status of the document with a call

```python
response = cogniswitch_agent_pack.run(
    "Tell me the status of https://cogniswitch.ai/developer"
)
```

### Use the cogniswitch agent for answering with a single call

```python
response = cogniswitch_agent_pack.run(
    "Answer the question- Tell me about cogniswitch"
)
```
