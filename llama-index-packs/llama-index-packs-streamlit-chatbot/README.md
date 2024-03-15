# Steamlit Chatbot Pack

Build a chatbot powered by LlamaIndex that augments an LLM with the contents of Snowflake's Wikipedia page (or your own data).

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message`
- Uses LlamaIndex to load and index data and create a chat engine that will retrieve context from that data to respond to each user query
- UI will stream each answer from the LLM

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
pip install llama-index
```

```bash
llamaindex-cli download-llamapack StreamlitChatPack --download-dir ./streamlit_chatbot_pack
```

You can then inspect the files at `./streamlit_chatbot_pack` and use them as a template for your own project!

To run the app directly, use in your terminal:

```bash
export OPENAI_API_KEY="sk-..."
streamlit run ./streamlit_chatbot_pack/base.py
```
