### Core Principles


- **Transparency**: highlight when and where you used AI to generate code, and explain how you verified and validated it
- **Accountability**: we require human oversight for every contribution, and we hold human developers accountable for their changes: in this sense, it is best if you don't propose changes you don't understand or cannot maintain
- **Quality**: AI code should meet the same quality standards as human code: this means being documented, tested, and following existing patterns


### Guidelines


**Use for**


- refactors of existing code, writing boilerplate or repetitive patterns, create tests
- improving existing documentation, or to write concise explanatory comments
- helpers and utilities


**Avoid for**


- complex code changes (without thoroughly reviewing what AI produced)
- core architectural changes
- excessively large code changes. Despite the fact that AI can create thousands of lines of code in a relatively small amount of time, reviewing large code changes takes much longer and much more energy from us maintainers
- creating code you don't understand or cannot maintain long-term
- repetitive, self-explanatory or excessively long comments, docstrings or documentation
- secrets handling or security-related code


Overall, our suggestion is to use AI by starting with **small changes**, validating often, making sure tests pass and quality criteria are met, and build incrementally.


---


## 👥 Join the Community


We’d love to hear from you and collaborate! Join our Discord community to ask questions, share ideas, or just chat with fellow developers.


Join us on Discord <https://discord.gg/dGcwcsnxhU>


---


## 🌟 Acknowledgements


Thank you for considering contributing to LlamaIndex! Every contribution—whether it’s code, documentation, or ideas—helps make this project better for everyone.


Happy coding! 😊




---


## 2 — Contribute a pack, reader, tool, or dataset (formerly from Llama Hub)


New integration packages are no longer accepted in this monorepo. Please maintain new integrations in standalone repositories and publish them to PyPI independently.


If you want an external integration to be considered for discovery on [LlamaHub](https://llamahub.ai/), use the [Feature Request form](https://github.com/run-llama/llama_index/issues/new?assignees=&labels=enhancement%2Ctriage&projects=&template=feature-form.yml&title=%5BFeature+Request%5D%3A+). Include:


- the package name and PyPI URL
- the source repository and documentation URL
- the integration type (reader, retriever, tool, pack, or dataset)
- the supported LlamaIndex version and a minimal usage example


The historical [`run-llama/llama-hub`](https://github.com/run-llama/llama-hub) repository is archived and read-only; its old pull-request workflow is retained for historical reference only.

