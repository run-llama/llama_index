
# LlamaIndex Multimodal RAG with Guardrails

## Overview

<img src="https://github.com/vntuananhbui/llama_index/blob/vntuananhbui-multimodal-rag-guardrail/docs/docs/examples/multimodal-rag-guardrail-gemini/Guardrail_data_workflow-4.png">

This project enhances the Multimodal Retrieval-Augmented Generation (RAG) example in **LlamaIndex** by introducing **LLM Guard** guardrails. These guardrails ensure safer interactions by detecting and mitigating toxic, sensitive, or inappropriate content in both input queries and output responses.

Key contributions include:
- **Input Scanners**: Validate user queries before processing.
- **Output Scanners**: Ensure generated responses comply with safety standards.
- **Custom Query Engine**: Seamlessly integrates input and output guardrails into the query execution flow.

---

## Features

### 1. Input Guardrails
- Detects risks such as toxicity, sensitive content, or custom-defined threats in user queries.
- Blocks or sanitizes unsafe queries before proceeding with retrieval or generation.

### 2. Output Guardrails
- Evaluates generated responses for compliance with predefined safety rules.
- Sanitizes or blocks unsafe responses before returning them to the user.

### 3. Custom Query Engine
- Extends the `CustomQueryEngine` in LlamaIndex to support input and output guardrails.
- Provides detailed metadata for triggered guardrails.

---

## Installation

### Prerequisites
- Python 3.8 or later
- Dependencies: LlamaIndex, LLM Guard

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/llamaindex-guardrails
   cd llamaindex-guardrails
   ```

2. Ensure you have access to an LLM (e.g., Gemini-1.5-flash) and configure API keys as required.

---

## Usage

### Adding Scanners
Scanners are the core of the guardrail system, checking inputs and outputs for specific risks.

#### Example: Toxicity Scanner for Input
Link: https://llm-guard.com/input_scanners/toxicity/
```python
from llm_guard.input_scanners import Toxicity


##Here is the rail output for all scanner
def result_response(
    guardrail_type,
    activated,
    guard_output,
    is_valid,
    risk_score,
    threshold,
    response_text,
):
    """
    Standardizes the result format for all guardrail checks.
    """
    return {
        "guardrail_type": guardrail_type,
        "activated": activated,
        "guardrail_detail": {
            "guard_output": guard_output,
            "is_valid": is_valid,
            "risk_score/threshold": f"{risk_score}/{threshold}",
            "response_text": response_text,
        },
    }


def guardrail_toxicLanguage(prompt):
    # Interact with the LLM to generate a response
    print(f"Prompt: {prompt}")

    # Generate the response using the LLM (Gemini-1.5-flash)
    response = completion(
        model="gemini/gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content

    # Define the threshold and scan for toxicity
    threshold = 0.5
    toxic_scanner = Toxicity(threshold=threshold, match_type=MatchType.FULL)
    sanitized_output, is_valid, risk_score = toxic_scanner.scan(prompt)

    return result_response(
        guardrail_type="Toxicity",
        activated=not is_valid,
        guard_output=sanitized_output,
        is_valid=is_valid,
        risk_score=risk_score,
        threshold=threshold,
        response_text=response_text,
    )
```

#### Adding to Input Scanner List
Pass the custom scanner as part of the `input_scanners` list when initializing the query engine.

### Custom Query Engine
The `MultimodalQueryEngine` processes the query while applying guardrails:

#### Initialization
```python
from llama_index.core.query_engine import MultimodalQueryEngine

query_engine = MultimodalQueryEngine(
    retriever=my_retriever,
    multi_modal_llm=my_multi_modal_llm,
    input_scanners=[guardrail_toxicLanguage],
    output_scanners=[my_output_scanner],
)
```

#### Querying the Engine
```python
response = query_engine.query("Your query here")
print(response)
```

---

## Extensibility

- **Custom Scanners**: Easily define and integrate new scanners for specific use cases.
- **Guardrail Types**: Extend beyond toxicity to include other checks like bias, harmful content, or compliance.
- **Metadata**: Utilize the detailed metadata for analytics or debugging.

---

## Contributing

We welcome contributions to make this project more robust and versatile! Feel free to submit issues or pull requests.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Acknowledgments

- **LlamaIndex**: For providing the foundational RAG framework.
- **LLM Guard**: For the tools to ensure AI safety.
