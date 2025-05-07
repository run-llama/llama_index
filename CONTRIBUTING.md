# 🚀 Contributing to LlamaIndex

Welcome to **LlamaIndex**! We’re excited that you want to contribute and become part of our growing community. Whether
you're interested in building integrations, fixing bugs, or adding exciting new features, we've made it easy for you to
get started.

---

## 🎯 Quick Start Guide

We use `uv` as the package and project manager for all the Python packages in this repository. Before contributing, make
sure you have `uv` installed. On macOS and Linux:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more install options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

If you're ready to dive in, here’s a quick setup guide to get you going:

1. **Fork** the GitHub repo, clone your fork and open a terminal at the root of the git repository `llama_index`.
2. At the root of the repo, run the following command to setup the global virtual environment we use for the
   pre-commit hooks and the linters:

```bash
uv sync
```

If you've changed documentation:

```bash
uv run make lint
```

3. Navigate to the project folder you want to work on. For example, if you want to work on the OpenAI llm integration:

```bash
cd llama-index-integrations/llms/llama-index-llms-openai
```

4. `uv` will take care of creating and setting up the virtual environment for the specific project you're working on.
   For example, to run the tests you can do:

```bash
uv run -- pytest
```

5. If you want to create the virtual environment explicitly, without `uv run`ning a command:

```bash
uv venv
```

6. To activate the virtual environment:

```bash
source .venv/bin/activate
```

**That’s it!** The package you're working on is already installed in editable mode, so you can go on, change the code
and run the tests!

Once you get familiar with the project, scroll down to the [Development Guidelines](#-Development-Guidelines) for more
details.

---

## 🛠️ What Can You Work On?

There’s plenty of ways to contribute—whether you’re a seasoned Python developer or just starting out, your contributions are welcome! Here are some ideas:

### 1. 🆕 Extend Core Modules

Help us extend LlamaIndex's functionality by contributing to any of our core modules. Think of this as unlocking new superpowers for LlamaIndex!

- **New Integrations** (e.g., connecting new LLMs, storage systems, or data sources)
- **Data Loaders**, **Vector Stores**, and more!

Explore the different modules below to get inspired!

New integrations should meaningfully integrate with existing LlamaIndex framework components. At the discretion of LlamaIndex maintainers, some integrations may be declined.

### 2. 📦 Contribute Tools, Readers, Packs, or Datasets

Create new Packs, Readers, or Tools that simplify how others use LlamaIndex with various platforms.

### 3. 🧠 Add New Features

Have an idea for a feature that could make LlamaIndex even better? Go for it! We love innovative contributions.

### 4. 🐛 Fix Bugs

Fixing bugs is a great way to start contributing. Head over to our [Github Issues](https://github.com/run-llama/llama_index/issues) page and find bugs tagged as [`good first issue`](https://github.com/run-llama/llama_index/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

### 5. 🎉 Share Usage Examples

If you’ve used LlamaIndex in a unique or creative way, consider sharing guides or notebooks. This helps other developers learn from your experience.

### 6. 🧪 Experiment

Got an out-there idea? We’re open to experimental features—test it out and make a PR!

### 7. 📄 Improve Documentation & Code Quality

Help make the project easier to navigate by refining the docs or cleaning up the codebase. Every improvement counts!

---

## 🔥 How to Extend LlamaIndex’s Core Modules

### Data Loaders

A **data loader** ingests data from any source and converts it into `Document` objects that LlamaIndex can parse and index.

- **Interface**:
  - `load_data`: Returns a list of `Document` objects.
  - `lazy_load_data`: Returns an iterable of `Document` objects (useful for large datasets).

**Example**: [MongoDB Reader](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-mongodb)

💡 **Ideas**: Want to load data from a source not yet supported? Build a new data loader and submit a PR!

### Node Parsers

A **node parser** converts `Document` objects into `Node` objects—atomic chunks of data that LlamaIndex works with.

- **Interface**:
  - `get_nodes_from_documents`: Returns a list of `Node` objects.

**Example**: [Hierarchical Node Parser](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/relational/hierarchical.py)

💡 **Ideas**: Add new ways to structure hierarchical relationships in documents, like play-act-scene or chapter-section formats.

### Text Splitters

A **text splitter** breaks down large text blocks into smaller chunks—this is key for working with LLMs that have limited context windows.

- **Interface**:
  - `split_text`: Takes a string and returns smaller strings (chunks).

**Example**: [Token Text Splitter](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/token.py)

💡 **Ideas**: Build specialized text splitters for different content types, like code, dialogues, or dense data!

### Vector Stores

Store embeddings and retrieve them via similarity search with **vector stores**.

- **Interface**:
  - `add`, `delete`, `query`, `get_nodes`, `delete_nodes`, `clear`

**Example**: [Pinecone Vector Store](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone)

💡 **Ideas**: Create support for vector databases that aren't yet integrated!

### Query Engines & Retrievers

- **Query Engines** implement `query` to return structured responses.
- **Retrievers** retrieve relevant nodes based on queries.

💡 **Ideas**: Design fancy query engines that combine retrievers or add intelligent processing layers!

---

## ✨ Steps to Contribute

1. **Fork** the repository on GitHub.
2. **Clone** your fork to your local machine.
   ```bash
   git clone https://github.com/your-username/llama_index.git
   ```
3. **Create a branch** for your work.
   ```bash
   git checkout -b your-feature-branch
   ```
4. **Set up your environment** (follow the [Quick Start Guide](#quick-start-guide)).
5. **Work on your feature or bugfix**, ensuring you have unit tests covering your code.
6. **Commit** your changes, then push them to your fork.
   ```bash
   git push origin your-feature-branch
   ```
7. **Open a pull request** on GitHub.

And voilà—your contribution is ready for review!

---

## 🧑‍💻 Development Guidelines

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

By default, CICD will fail if test coverage is less than 50% -- so please do add tests for your code!

---

## 👥 Join the Community

We’d love to hear from you and collaborate! Join our Discord community to ask questions, share ideas, or just chat with fellow developers.

Join us on Discord <https://discord.gg/dGcwcsnxhU>

---

## 🌟 Acknowledgements

Thank you for considering contributing to LlamaIndex! Every contribution—whether it’s code, documentation, or ideas—helps make this project better for everyone.

Happy coding! 😊
