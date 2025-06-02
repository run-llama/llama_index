# LlamaIndex Llms Integration: Clarifai

### Installation

```bash
%pip install llama-index-llms-clarifai
!pip install llama-index
!pip install clarifai
```

### Basic Usage

```py
# Set Clarifai PAT as an environment variable
import os

os.environ["CLARIFAI_PAT"] = "<YOUR CLARIFAI PAT>"

# Import Clarifai package
from llama_index.llms.clarifai import Clarifai

# Example parameters
params = dict(
    user_id="clarifai",
    app_id="ml",
    model_name="llama2-7b-alternative-4k",
    model_url="https://clarifai.com/clarifai/ml/models/llama2-7b-alternative-4k",
)

# Initialize the LLM
# Method:1 using model_url parameter
llm_model = Clarifai(model_url=params["model_url"])

# Method:2 using model_name, app_id & user_id parameters
llm_model = Clarifai(
    model_name=params["model_name"],
    app_id=params["app_id"],
    user_id=params["user_id"],
)

# Call complete function
llm_response = llm_model.complete(
    prompt="write a 10 line rhyming poem about science"
)
print(llm_response)

# Output
# Science is fun, it's true!
# From atoms to galaxies, it's all new!
# With experiments and tests, we learn so fast,
# And discoveries come from the past.
# It helps us understand the world around,
# And makes our lives more profound.
# So let's embrace this wondrous art,
# And see where it takes us in the start!
```

### Call chat function

```py
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="write about climate change in 50 lines")
]
response = llm_model.chat(messages)
print(response)

# Output
# user: or less.
# Climate change is a serious threat to our planet and its inhabitants.
# Rising temperatures are causing extreme weather events, such as hurricanes,
# droughts, and wildfires. Sea levels are rising, threatening coastal
# communities and ecosystems. The melting of polar ice caps is disrupting
# global navigation and commerce. Climate change is also exacerbating air
# pollution, which can lead to respiratory problems and other health issues.
# It's essential that we take action now to reduce greenhouse gas emissions
# and transition to renewable energy sources to mitigate the worst effects
# of climate change.
```

### Using Inference parameters

```py
# Alternatively, you can call models with inference parameters.

# Here is an inference parameter example for GPT model.
inference_params = dict(temperature=str(0.3), max_tokens=20)

llm_response = llm_model.complete(
    prompt="What is nuclear fission and fusion?",
    inference_params=inference_params,
)

messages = [ChatMessage(role="user", content="Explain about the big bang")]
response = llm_model.chat(messages, inference_params=inference_params)

print(response)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/clarifai/
