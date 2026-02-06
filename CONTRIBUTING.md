# Contributing to LlamaIndex

Welcome to **LlamaIndex**! We’re excited that you want to contribute and become part of our growing community. Whether
you're interested in building integrations, fixing bugs, or adding exciting new features, we've made it easy for you to
get started.

---

## Quick Start Guide

We use `uv` as the package and project manager for all the Python packages in this repository. Before contributing, make sure you have `uv` installed (see [installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

If you're ready to dive in, here’s a quick setup guide to get you going:

1. **Fork** the GitHub repo, clone your fork and open a terminal at the root of the git repository `llama_index`.
2. At the root of the repo, run the following command to setup the global virtual environment we use for the pre-commit hooks and the linters:

```bash
uv sync
```

Install `pre-commit` to run pre-commit hooks on each commit:

```bash
uv run pre-commit install
```

Whenever you make changes, make sure they comply with linting rules:

```bash
uv run make lint
```

3. Navigate to the project folder you want to work on. For example, if you want to work on the OpenAI LLM integration:

```bash
cd llama-index-integrations/llms/llama-index-llms-openai
```

4. `uv` will take care of creating and setting up the virtual environment for the specific project you're working on. For example, to run the tests you can do:

```bash
uv run -- pytest
```

**That’s it!** The package you're working on is already installed in editable mode, so you can go on, change the code and run the tests!

Once you get familiar with the project, scroll down to the [Development Guidelines](#-Development-Guidelines) for more details.

---

## What can you work on?

We suggest working on:

- Core modules (`llama-index-core` and `llama-index-instrumentation`), contributing with refactoring, bug fixes and extensions
- Documentation (`docs`), helping us improve our current docs and keep them updated.
- Main integrations, such as `llama-index-llms`, `llama-index-embeddings` or `llama-index-vector-stores`, providing help with maintaining them
- New integrations with third party services

While we welcome contributions, we do not recommend to work on these areas:

- Experimental features (`llama-index-experimental`)
- Packs (`llama-index-packs`)
- Finentuning (`llama-index-finentuning`)
- CLI (`llama-index-cli`)

---

## Steps to Contribute

1. **Fork** the repository on GitHub.
2. **Clone** your fork to your local machine.
   ```bash
   git clone https://github.com/your-username/llama_index.git
   ```
3. **Create a branch** for your work.
   ```bash
   git checkout -b your-feature-branch
   ```
4. **Set up your environment** (follow the [Quick Start Guide](#-quick-start-guide)).
5. **Work on your feature or bugfix**, ensuring you have unit tests covering your code.
6. **Commit** your changes, then push them to your fork.
   ```bash
   git push origin your-feature-branch
   ```
7. **Open a pull request** on GitHub.

---

## Development Guidelines

### Repo Structure

LlamaIndex is organized as a **monorepo**, meaning different packages live within this single repository. You can focus on a specific package depending on your contribution:

- **Core package**: [`llama-index-core/`](https://github.com/run-llama/llama_index/tree/main/llama-index-core)
- **Integrations**: e.g., [`llama-index-integrations/`](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations)

### Running Tests

We use `pytest` for testing. Make sure you run tests in each package you modify:

```bash
uv run -- pytest
```

If you’re integrating with a remote system, **mock** it to prevent test failures from external changes.

By default, CI/CD will fail if test coverage is less than 50%, so make sure your packages or changes are covered by tests.

---

## How to Use AI when Contributing

We welcome AI-assisted contributions, but we ask you to follow some core principles and guidelines that can help make the contribution and review process smoother for both you and us maintainers.

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

Overall, our suggestion is use AI by starting with **small changes**, validating often, making sure tests pass and quality criteria are met, and build incrementally.

---

## 👥 Join the Community

We’d love to hear from you and collaborate! Join our Discord community to ask questions, share ideas, or just chat with fellow developers.

Join us on Discord <https://discord.gg/dGcwcsnxhU>

---

## 🌟 Acknowledgements

Thank you for considering contributing to LlamaIndex! Every contribution—whether it’s code, documentation, or ideas—helps make this project better for everyone.

Happy coding! 😊
