from llama_index.tools.hive import HiveToolSpec, HiveSearchMessage

hive_tool = HiveToolSpec(api_key="add_your_hive_api_key")

# Simple prompt query
results = hive_tool.search(
    prompt="What is the current price of Ethereum?",
    include_data_sources=True
)

print("results ", results)

# Chat-style conversation
chat_msgs = [
    HiveSearchMessage(role="user", content="Price of what?"),
    HiveSearchMessage(role="assistant", content="Please specify asset."),
    HiveSearchMessage(role="user", content="BTC"),
]
results = hive_tool.search(
    messages=chat_msgs,
    include_data_sources=True
)

print("results ", results)
