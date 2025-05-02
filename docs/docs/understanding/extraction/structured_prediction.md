# Structured Prediction

Structured Prediction gives you more granular control over how your application calls the LLM and uses Pydantic. We will use the same `Invoice` class, load the PDF as we did in the previous example, and use OpenAI as before. Instead of creating a structured LLM, we will call `structured_predict` on the LLM itself; this a method of every LLM class.

Structured predict takes a Pydantic class and a Prompt Template as arguments, along with keyword arguments of any variables in the prompt template.

```python
from llama_index.core.prompts import PromptTemplate

prompt = PromptTemplate(
    "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
)

response = llm.structured_predict(
    Invoice, prompt, text=text, company_name="Uber"
)
```

As you can see, this allows us to include additional prompt direction for what the LLM should do if Pydantic isn’t quite enough to parse the data correctly. The response object in this case is the Pydantic object itself. We can get the output as JSON if we want:

```python
json_output = response.model_dump_json()
print(json.dumps(json.loads(json_output), indent=2))
```

```python
{
    "invoice_id": "Uber-2024-10-10",
    "date": "2024-10-10T19:49:00",
    "line_items": [
        {"item_name": "Trip fare", "price": 12.18},
        {"item_name": "Access for All Fee", "price": 0.1},
        ...,
    ],
}
```

`structured_predict` has several variants available for different use-cases include async (`astructured_predict`) and streaming (`stream_structured_predict`, `astream_structured_predict`).

## Under the hood

Depending on which LLM you use, `structured_predict` is using one of two different classes to handle calling the LLM and parsing the output.

### FunctionCallingProgram

If the LLM you are using has a function calling API, `FunctionCallingProgram` will

- Convert the Pydantic object into a tool
- Prompts the LLM while forcing it to use this tool
- Returns the Pydantic object generated

This is generally a more reliable method and will be used by preference if available. However, some LLMs are text-only and they will use the other method.

### LLMTextCompletionProgram

If the LLM is text-only, `LLMTextCompletionProgram` will

- Output the Pydantic schema as JSON
- Send the schema and the data to the LLM with prompt instructions to respond in a form the conforms to the schema
- Call `model_validate_json()` on the Pydantic object, passing in the raw text returned from the LLM

This is notably less reliable, but supported by all text-based LLMs.

## Calling prediction classes directly

In practice `structured_predict` should work well for any LLM, but if you need lower-level control it is possible to call `FunctionCallingProgram` and `LLMTextCompletionProgram` directly and further customize what’s happening:

```python
textCompletion = LLMTextCompletionProgram.from_defaults(
    output_cls=Invoice,
    llm=llm,
    prompt=PromptTemplate(
        "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
    ),
)

output = textCompletion(company_name="Uber", text=text)
```

The above is identical to calling `structured_predict` on an LLM without function calling APIs and returns a Pydantic object just like `structured_predict` does. However, you can customize how the output is parsed by subclassing the `PydanticOutputParser`:

```python
from llama_index.core.output_parsers import PydanticOutputParser


class MyOutputParser(PydanticOutputParser):
    def get_pydantic_object(self, text: str):
        # do something more clever than this
        return self.output_parser.model_validate_json(text)


textCompletion = LLMTextCompletionProgram.from_defaults(
    llm=llm,
    prompt=PromptTemplate(
        "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
    ),
    output_parser=MyOutputParser(output_cls=Invoice),
)
```

This is useful if you are using a low-powered LLM that needs help with the parsing.

In the final section we will take a look at even [lower-level calls to the extract structured data](lower_level.md), including extracting multiple structures in the same call.
