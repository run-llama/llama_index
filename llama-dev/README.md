# llama-dev

The official CLI for development, testing, and automation in the LlamaIndex monorepo.

## Overview

`llama-dev` is a command-line tool designed to simplify development and testing workflows in the LlamaIndex monorepo. It provides commands for managing packages, running tests, and automating common development tasks.

## Installation

First step, create a virtual environment for the project:

```bash
uv venv
source .venv/bin/activate
```

Then install `llama_dev` in the virtual environment:

```bash
uv pip install -e .
```

You should now have the `llama-dev` CLI available in your path.

## Usage

> [!TIP]
> It's easier to run `llama-dev` from the root of the `llama_index` repo,
> otherwise you have to pass the `--repo-root` option at every command.

```bash
llama-dev --help
```

### Package Management

Get information about packages in the monorepo:

```bash
# Get info for a specific package
llama-dev pkg info llama-index-core

# Get info for all packages
llama-dev pkg info --all
```

Execute commands in package directories:

```bash
# Run a command in a specific package
llama-dev pkg exec --cmd "uv sync" llama-index-core

# Run a command in all packages
llama-dev pkg exec --cmd "uv sync" --all

# Same but exit at the first error
llama-dev pkg exec --cmd "uv" --all --fail-fast
```

### Testing

Run tests across the monorepo:

```bash
# Run tests for packages changed compared to main
llama-dev test --base-ref main

# Run tests with coverage
llama-dev test --base-ref main --cov

# Run tests with coverage threshold
llama-dev test --base-ref main --cov --cov-fail-under 80

# Adjust the number of parallel workers
llama-dev test --base-ref main --workers 4

# Run tests with fail-fast option
llama-dev test --base-ref main --fail-fast
```

## Features

- **Package Management**: Get information about packages and execute commands across multiple packages.
- **Smart Testing**: Automatically detects changed packages and their dependents, only running tests where needed.
- **Test Coverage**: Analyze test coverage and enforce minimum coverage thresholds.
- **Python Compatibility**: Skip packages incompatible with the current Python version.
- **Parallel Testing**: Run tests across multiple packages in parallel.

## Requirements

- Python 3.10+
- UV package manager (`uv`)
- Git

## License

This project is part of the LlamaIndex monorepo and follows its licensing.
