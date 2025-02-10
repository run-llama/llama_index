# Jira Integration

This tool integrates with Jira and enables fetching Jira issue details, comments, projects, and performing searches. You can query issues, retrieve specific issue details, or search for issues matching a keyword.

## Usage

Here is the example usage:

```python
from llama_index.tools.jira import JiraToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = JiraToolSpec(server_url=SERVER, email=EMAIL, api_token=API_KEY)

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

# Fetch a specific Jira issue by key
response = agent.chat(
    "Fetch Jira issue with the key 'PROJ-5' and give me the details."
)
print(response)

# Search for issues containing a specific keyword
response = agent.chat("Search for Jira issues containing 'login bug'.")
print(response)

# Fetch all Jira projects
response = agent.chat("List all Jira projects.")
print(response)

# Fetch all comments for a specific issue
response = agent.chat("Fetch comments for Jira issue 'PROJ-5'.")
print(response)
```
