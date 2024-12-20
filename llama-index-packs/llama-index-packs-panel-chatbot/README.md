# 🦙 Panel ChatBot Pack

Build a chatbot to talk to your Github repository.

Powered by LlamaIndex, OpenAI ChatGPT and [HoloViz Panel](https://panel.holoviz.org/reference/chat/ChatInterface.html).

![Panel Chat Bot](https://raw.githubusercontent.com/run-llama/llama-hub/main/llama_hub/llama_packs/panel_chatbot/panel_chatbot.png)

## 💁‍♀️ Explanation

This template

- Downloads and indexes a Github repository using the `llama_index` [`GithubRepositoryReader`](https://llamahub.ai/l/github_repo). The default repository is [holoviz/panel](https://github.com/holoviz/panel).
- Creates a [VectorStoreIndex](https://docs.llamaindex.ai/en/stable/changes/deprecated_terms.html#VectorStoreIndex) powered chat engine that will retrieve context from that data to respond to each user query.
- Creates a Panel [`ChatInterface`](https://panel.holoviz.org/reference/chat/ChatInterface.html) UI that will stream each answer from the chat engine.

## 🖥️ CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` Python package:

```bash
pip install llama-index
llamaindex-cli download-llamapack PanelChatPack --download-dir ./panel_chat_pack
```

You can then inspect the files in the `panel_chat_pack` folder and use them as a template for your own project!

To run the app directly, use in your terminal:

```bash
export OPENAI_API_KEY="sk-..."
export GITHUB_TOKEN='...'
panel serve ./panel_chat_pack/base.py
```

As an alternative to `panel serve`, you can run

```bash
python ./panel_chat_pack/base.py
```

## 🎓 Learn More

- [`GithubRepositoryReader`](https://llamahub.ai/l/github_repo)
- [`VectorStoreIndex`](https://docs.llamaindex.ai/en/stable/changes/deprecated_terms.html#VectorStoreIndex)
- [Panel Chat Components](https://panel.holoviz.org/reference/index.html#chat)
- [Panel Chat Examples](https://github.com/holoviz-topics/panel-chat-examples)

## 👍 Credits

- [Marc Skov Madsen](https://twitter.com/MarcSkovMadsen) for creating the template
- [Sophia Yang](https://twitter.com/sophiamyang) for creating the cute LLama image.

## 📈 Potential Improvements

- [ ] Improved Multi-user support
  - [ ] Loading queue: Users should not be able to download the same repository at the same time.
- [ ] Service Context
  - [ ] Enable users to define the service context including `model`, `temperature` etc.
- [ ] Better loading experience
  - [ ] Let the chat assistant show (more fine-grained) status messages. And provide more status changes
- [ ] Focus on the streaming text
  - [ ] The streaming text is not always in focus. I believe its a matter of adjusting the `auto_scroll_limit` limit.
- [ ] Fix minor CSS issues
  - [ ] See the `CSS_FIXES_TO_BE_UPSTREAMED_TO_PANEL` variable in the code
- [ ] Repo Manager
  - [ ] Support using multiple repos. For example the full HoloViz suite
