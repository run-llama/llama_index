# Putting It All Together

Congratulations! You've loaded your data, indexed it, stored your index, and queried your index. Now you've got to ship something to production. We can show you how to do that!

- In [Q&A Patterns](q_and_a/index.md) we'll go into some of the more advanced and subtle ways you can build a query engine beyond the basics.
  - The [terms definition tutorial](q_and_a/terms_definitions_tutorial.md) is a detailed, step-by-step tutorial on creating a subtle query application including defining your prompts and supporting images as input.
  - We have a guide to [creating a unified query framework over your indexes](../../examples/retrievers/reciprocal_rerank_fusion.ipynb) which shows you how to run queries across multiple indexes.
  - And also over [structured data like SQL](./structured_data/index.md)
- We have a guide on [how to build a chatbot](./chatbots/building_a_chatbot.md)
- We talk about [building agents in LlamaIndex](./agents.md)
- We have a complete guide to using [property graphs for indexing and retrieval](../../module_guides/indexing/lpg_index_guide.md)
- And last but not least we show you how to build [a full stack web application](./apps/index.md) using LlamaIndex

LlamaIndex also provides some tools / project templates to help you build a full-stack template. For instance, [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama) spins up a full-stack scaffold for you.

Check out our [Full-Stack Projects](../../community/full_stack_projects.md) page for more details.

We also have the [`llamaindex-cli rag` CLI tool](../../getting_started/starter_tools/rag_cli.md) that combines some of the above concepts into an easy to use tool for chatting with files from your terminal!
