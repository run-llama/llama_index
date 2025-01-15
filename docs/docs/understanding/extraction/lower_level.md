# Low-level structured data extraction

If your LLM supports tool calling and you need more direct control over how LlamaIndex extracts data, you can use `chat_with_tools` on an LLM directly. If your LLM does not support tool calling you can instruct your LLM directly and parse the output yourself. We’ll show how to do both.

## Calling tools directly

```python
from llama_index.core.program.function_program import get_function_tool

tool = get_function_tool(Invoice)

resp = llm.chat_with_tools(
    [tool],
    # chat_history=chat_history,  # can optionally pass in chat history instead of user_msg
    user_msg="Extract an invoice from the following text: " + text,
    # tool_choice="Invoice",  # can optionally force the tool call
)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_calls=False
)

outputs = []
for tool_call in tool_calls:
    if tool_call.tool_name == "Invoice":
        outputs.append(Invoice(**tool_call.tool_kwargs))

# use your outputs
print(outputs[0])
```

This is identical to `structured_predict` if the LLM has a tool calling API. However, if the LLM supports it you can optionally allow multiple tool calls. This has the effect of extracting multiple objects from the same input, as in this example:

```python
from llama_index.core.program.function_program import get_function_tool

tool = get_function_tool(LineItem)

resp = llm.chat_with_tools(
    [tool],
    user_msg="Extract line items from the following text: " + text,
    allow_parallel_tool_calls=True,
)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_calls=False
)

outputs = []
for tool_call in tool_calls:
    if tool_call.tool_name == "LineItem":
        outputs.append(LineItem(**tool_call.tool_kwargs))

# use your outputs
print(outputs)
```

If extracting multiple Pydantic objects from a single LLM call is your goal, this is how to do that.

## Direct prompting

If for some reason none of LlamaIndex’s attempts to make extraction easier are working for you, you can dispense with them and prompt the LLM directly and parse the output yourself, as here:

```python
schema = Invoice.model_json_schema()
prompt = "Here is a JSON schema for an invoice: " + json.dumps(
    schema, indent=2
)
prompt += (
    """
  Extract an invoice from the following text.
  Format your output as a JSON object according to the schema above.
  Do not include any other text than the JSON object.
  Omit any markdown formatting. Do not include any preamble or explanation.
"""
    + text
)

response = llm.complete(prompt)

print(response)

invoice = Invoice.model_validate_json(response.text)

pprint(invoice)
```

Congratulations! You have learned everything there is to know about structured data extraction in LlamaIndex.

## Other Guides

For a deeper look at structured data extraction with LlamaIndex, check out the following guides:

- [Structured Outputs](../../module_guides/querying/structured_outputs/index.md)
- [Pydantic Programs](../../module_guides/querying/structured_outputs/pydantic_program.md)
- [Output Parsing](../../module_guides/querying/structured_outputs/output_parser.md)
