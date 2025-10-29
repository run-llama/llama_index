# Jira Issue Tool

This tool performs basic Jira issue operations, including search, creation, commentation, deletion, and modifications of summary, assignee, task status, and due date.

## Usage

Example:

```python
from llama_index.tools.jira_issue import JiraIssueToolSpec
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)

tool_spec = JiraIssueToolSpec(
    server_url=SERVER, email=EMAIL, api_token=API_KEY
)

agent_name = "Jira Issue Agent"
project_key = "YOUR_PROJECT_KEY"
system_prompt = f"""You are a helpful assistant that answers JIRA-related questions with a list of JIRA tools available for use. Your project key is '{project_key}'."""

agent = ReActAgent.from_tools(
    jira_tools,
    llm=llm,
    callback_manager=callback_manager,
    name=agent_name,
    system_prompt=system_prompt,
    verbose=True,
    max_iterations=20,
)

# Multi-step bug issue creation
response = await agent.achat(
    "Create a test bug due by Aug 2 2025 and assign it to John Doe. this bug addresses the wrong temperature setting in the LLMs."
)
print(response)

# Search & Reassign an issue
response = await agent.achat(
    "There's an issue about building a jira agent, assigned to John Doe. Find it and reassign it to Jane Doe."
)
print(response)

# Search & Mark an issue 'Done'
response = await agent.achat(
    "Find an issue about llamaindex due on Aug 2 2025. Change its status to Done."
)
print(response)

# Delete issues
response = await agent.achat("Delete all issues assigned to John Doe")
print(response)
```
