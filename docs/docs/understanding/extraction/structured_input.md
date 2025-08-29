# Structured Input

The other side of structured data, beyond the output, is the input: many prompting guides and best practices, indeed, include some techniques such as XML tagging of the input prompt to boost the LLM's understanding of the input.

LlamaIndex offers you the possibility of natively formatting your inputs as XML snippets, leveraging [banks](https://masci.github.io/banks) and [Jinja](https://jinja.palletsprojects.com/en/stable/) (make sure to have `llama-index>=0.12.34` installed).

## Using Structured Input Alone

Here is a simple example of how to use structured inputs with Pydantic models:

```python
from pydantic import BaseModel
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI
from typing import Dict

template_str = "Please extract from the following XML code the contact details of the user:\n\n```xml\n{{ user | to_xml }}\n```\n\n"
prompt = RichPromptTemplate(template_str)


class User(BaseModel):
    name: str
    surname: str
    age: int
    email: str
    phone: str
    social_accounts: Dict[str, str]


user = User(
    name="John",
    surname="Doe",
    age=30,
    email="john.doe@example.com",
    phone="123-456-7890",
    social_accounts={"bluesky": "john.doe", "instagram": "johndoe1234"},
)

## check how the prompt would look like

prompt.format(user=user)

llm = OpenAI()

response = llm.chat(prompt.format_messages(user=user))

print(response.message.content)
```

As you can see, in order to employ the structured output, we need to use a Jinja expression (delimited by `{{}}`) with the `to_xml` filter (the filtering operator is `|`).

## Combining Structured Input with Structured Output

The combination of structured input and structured output can really boost the consistency (and thus reliability) of your LLM's output.

With this code snippet below, you can see how you can chain these two step of data structuring.

```python
from pydantic import Field
from typing import Optional


class SocialAccounts(BaseModel):
    instagram: Optional[str] = Field(default=None)
    bluesky: Optional[str] = Field(default=None)
    x: Optional[str] = Field(default=None)
    mastodon: Optional[str] = Field(default=None)


class ContactDetails(BaseModel):
    email: str
    phone: str
    social_accounts: SocialAccounts


sllm = llm.as_structured_llm(ContactDetails)

structured_response = await sllm.achat(prompt.format_messages(user=user))

print(structured_response.raw.email)
print(structured_response.raw.phone)
print(structured_response.raw.social_accounts.instagram)
print(structured_response.raw.social_accounts.bluesky)
```

If you want a more in-depth guide to structured input, check out this [example notebook](https://docs.llamaindex.ai/en/latest/examples/prompts/structured_input).
