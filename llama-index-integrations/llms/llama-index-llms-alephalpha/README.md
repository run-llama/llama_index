# LlamaIndex LLM Integration: Aleph Alpha

This README details the process of integrating Aleph Alpha's Large Language Models (LLMs) with LlamaIndex. Utilizing Aleph Alpha's API, users can generate completions, facilitate question-answering, and perform a variety of other natural language processing tasks directly within the LlamaIndex framework.

## Features

- **Text Completion:** Use Aleph Alpha LLMs to generate text completions for prompts.
- **Model Selection:** Access the latest Aleph Alpha models, including the Luminous model family, to generate responses.
- **Advanced Sampling Controls:** Customize the response generation with parameters like temperature, top_k, top_p, presence_penalty, and more, to fine-tune the creativity and relevance of the generated text.
- **Control Parameters:** Apply attention control parameters for advanced use cases, affecting how the model focuses on different parts of the input.

## Installation

```bash
pip install llama-index-llms-alephalpha
```

## Usage

```python
from llama_index.llms.alephalpha import AlephAlpha
```

1. **Request Parameters:**

   - `model`: Specify the model name (e.g., `luminous-base-control`). The latest model version is always used.
   - `prompt`: The text prompt for the model to complete.
   - `maximum_tokens`: The maximum number of tokens to generate.
   - `temperature`: Adjusts the randomness of the completions.
   - `top_k`: Limits the sampled tokens to the top k probabilities.
   - `top_p`: Limits the sampled tokens to the cumulative probability of the top tokens.
   - `log_probs`: Set to `true` to return the log probabilities of the tokens.
   - `echo`: Set to `true` to return the input prompt along with the completion.
   - `penalty_exceptions`: A list of tokens that should not be penalized.
   - `n`: Number of completions to generate.

2. **Advanced Sampling Parameters:** (Optional)

   - `presence_penalty` & `frequency_penalty`: Adjust to discourage repetition.
   - `sequence_penalty`: Reduces likelihood of repeating token sequences.
   - `hosting`: Option to process the request in Aleph Alpha's own datacenters for enhanced data privacy.

## Response Structure

    * `model_version`: The name and version of the model used.
    * `completions`: A list containing the generated text completion(s) and optional metadata:
        * `completion`: The generated text completion.
        * `log_probs`: Log probabilities of the tokens in the completion.
        * `raw_completion`: The raw completion without any post-processing.
        * `completion_tokens`: Completion split into tokens.
        * `finish_reason`: Reason for completion termination.
    * `num_tokens_prompt_total`: Total number of tokens in the input prompt.
    * `num_tokens_generated`: Number of tokens generated in the completion.

## Example

Refer to the [example notebook](../../../docs/examples/llm/alephalpha.ipynb) for a comprehensive guide on generating text completions with Aleph Alpha models in LlamaIndex.

## API Documentation

For further details on the API and available models, please consult [Aleph Alpha's API Documentation](https://docs.aleph-alpha.com/api/complete/).
