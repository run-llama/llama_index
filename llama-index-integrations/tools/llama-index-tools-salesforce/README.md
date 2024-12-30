# Salesforce Tool

This tool connects to a Salesforce environment and allow the Agent to perform SOQL and SOSL queries.

## Usage

This tool is a wrapper tool using the simple salesforce library. More information on this library [here](https://simple-salesforce.readthedocs.io/)

Here's an example usage of the Salesforce Tool:

```python
from llama_index.tools.salesforce import SalesforceToolSpec

# Initialize the tool with your Salesforce credentials and other relevant details
sf = SalesforceToolSpec(
    username=sf_username,
    password=sf_password,
    consumer_key=sf_consumer_key,
    consumer_secret=sf_consumer_secret,
    domain="test",
)

agent = OpenAIAgent.from_tools(
    sf.to_tool_list(),
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
    memory=memory,
)

agent.chat("List 3 Accounts in Salesforce")
agent.chat("Provide information on a customer account John Doe")
```

`execute_sosl` - Returns the result of a Salesforce search as a dict decoded from the Salesforce response JSON payload.

`execute_soql` - Returns the full set of results for the `query`. The returned dict is the decoded JSON payload from the final call to Salesforce, but with the `totalSize` field representing the full number of results retrieved and the `records` list representing the full list of records retrieved.

This loader is designed to be used as a way to load data as a Tool in a Agent.
