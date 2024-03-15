## Cogniswitch ToolSpec

**Use CogniSwitch to build production ready applications that can consume, organize and retrieve knowledge flawlessly. Using the framework of your choice, in this case LlamaIndex, CogniSwitch helps alleviate the stress of decision making when it comes to, choosing the right storage and retrieval formats. It also eradicates reliability issues and hallucinations when it comes to responses that are generated. Get started by interacting with your knowledge in just three simple steps**

visit [https://www.cogniswitch.ai/developer](https://www.cogniswitch.ai/developer?utm_source=llamaindex&utm_medium=llamaindexbuild&utm_id=dev).

**Registration:**

- Signup with your email and verify your registration
- You will get a mail with a platform token and OAuth token for using the services.

**Step 1: Instantiate the Cogniswitch ToolSpec:**

- Use your Cogniswitch token, openAI API key, OAuth token to instantiate the toolspec.

**Step 2: Instantiate the Agent:**

- Instantiate the agent with the list of tools from the toolspec.

**Step 3: Cogniswitch Store data:**

- Make the call to the agent by giving the file path or url to the agent input.
- The agent will pick the tool and use the file/url and it will be processed and stored in your knowledge store.
- You can check the status of document processing with a call to the agent. Alternatively you can also check in [cogniswitch console](- You can check the status of document processing with a call to the agent. Alternatively you can also check in [cogniswitch console](https://console.cogniswitch.ai:8443/login?utm_source=llamaindex&utm_medium=llamaindexbuild&utm_id=dev).

**Step 4: Cogniswitch Answer:**

- Make the call to the agent by giving query as agent input.
- You will get the answer from your knowledge as the response.

### Import Required Libraries

```python
import warnings

warnings.filterwarnings("ignore")
import os
from llama_index.tools.cogniswitch import CogniswitchToolSpec
from llama_index.core.agent import ReActAgent
```

### Cogniswitch Credentials and OpenAI token

```python
# os.environ["OPENAI_API_KEY"] = <your openai token>
# cs_token = <your cogniswitch platform token>
# oauth_token = <your cogniswitch apikey>
```

### Instantiate the Tool Spec

```python
toolspec = CogniswitchToolSpec(cs_token=cs_token, apiKey=oauth_token)
```

### Get the list of tools

```python
tool_lst = toolspec.to_tool_list()
```

### Instantiate the agent with the tool list

```python
agent = ReActAgent.from_tools(tool_lst)
```

### Use the agent for storing data in cogniswitch with a single call

```python
store_response = agent.chat(
    """
                            https://cogniswitch.ai/developer
                            this site is about cogniswitch website for developers.
                           """
)
print(store_response)
```

    {'data': {'knowledgeSourceId': 43, 'sourceType': 'https://cogniswitch.ai/developer', 'sourceURL': None, 'sourceFileName': None, 'sourceName': 'Cogniswitch dev', 'sourceDescription': 'This is a cogniswitch website for developers.', 'status': 'UPLOADED'}, 'list': None, 'message': "We're processing your content & will send you an email on completion, hang tight!", 'statusCode': 1000}

### Use the agent to know the document status with a single call

```python
response = agent.chat("Tell me the status of Cogniswitch Developer Website")
```

```python
print(response)
```

    The document "Cogniswitch Developer Website" is currently being processed.

### Use the agent for answering a query with a single call

```python
answer_response = agent.chat("tell me about cogniswitch")
print(answer_response)
```

    {'data': {'answer': 'CogniSwitch is a technology platform that enhances the reliability of Generative AI applications for enterprises. It does this by gathering and organizing knowledge from documented sources, eliminating hallucinations and bias in AI responses. The platform uses AI to automatically gather and organize knowledge, which can then be reviewed and curated by experts before being published. The CogniSwitch API enables Gen AI applications to access this knowledge as needed, ensuring reliability. It is specifically designed to complement Generative AI and offers customized solutions for different business functions within an enterprise.'}, 'list': None, 'message': None, 'statusCode': 1000}

The tool is designed to store data and retrieve answers based on the knowledge provided. check out the [link](https://github.com/run-llama/llama-hub/blob/main/llama_hub/tools/notebooks/cogniswitch.ipynb) for examples.
