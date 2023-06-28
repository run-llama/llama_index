# 🔢 Structured Outputs

 The ability of LLMs to produce structured outputs are important for downstream applications that rely on reliably parsing output values. 
LlamaIndex itself also relies on structured output in the following ways.
- **Document retrieval**: Many data structures within LlamaIndex rely on LLM calls with a specific schema for Document retrieval. For instance, the tree index expects LLM calls to be in the format "ANSWER: (number)".
- **Response synthesis**: Users may expect that the final response contains some degree of structure (e.g. a JSON output, a formatted SQL query, etc.)

LlamaIndex provides a variety of modules enabling LLMs to produce outputs in a structured format. We provide modules at different levels of abstraction:
- **Output Parsers**: These are modules that operate before and after an LLM text completion endpoint. They are not used with LLM function calling endpoints (since those contain structured outputs out of the box).
- **Pydantic Programs**: These are generic modules that map an input prompt to a structured output, represented by a Pydantic object. They may use function calling APIs or text completion APIs + output parsers.
- **Pre-defined Pydantic Program**: We have pre-defined Pydantic programs that map inputs to specific output types (like dataframes).

See the sections below for an overview of output parsers and Pydantic programs.

## 🔬 Anatomy of a Structured Output Function

Here we describe the different components of an LLM-powered structured output function. The pipeline depends on whether you're using a **generic LLM text completion API** or an **LLM function calling API**>.

![](/_static/structured_output/diagram1.png)

With generic completion APIs, the inputs and outputs are handled by text prompts. The output parser plays a role before and after the LLM call in ensuring structured outputs. Before the LLM call, the output parser can
append format instructions to the prompt. After the LLM call, the output parser can parse the output to the specified instructions.

With function calling APIs, the output is inherently in a structured format, and the input can take in the signature of the desired object. The structured output just needs to be cast in the right object format (e.g. Pydantic).

## Output Parser Modules

```{toctree}
---
maxdepth: 2
---
output_parser.md
```

## Pydantic Program Modules

```{toctree}
---
maxdepth: 2
---
pydantic_program.md
```
