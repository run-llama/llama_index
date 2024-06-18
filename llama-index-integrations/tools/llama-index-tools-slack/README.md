# Slack Tool

This tool fetches the text from a list of Slack channels. You will need to initialize the loader with your Slack API Token or have the `SLACK_BOT_TOKEN` environment variable set.

## Usage

```python
from llama_index.tools.slack import SlackToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = SlackToolSpec(slack_token="token")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What is the most recent message in the announcements channel?")
```

`load_data`: Loads messages from a list of channels
`send_message`: Sends a message to a channel
`fetch_channel`: Fetches the list of channels

This loader is designed to be used as a way to load data as a Tool in a Agent.
