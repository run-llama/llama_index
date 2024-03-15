# LlamaIndex Llms Integration: Groq

Welcome to Groq! 🚀 At Groq, we've developed the world's first Language Processing Unit™, or LPU. The Groq LPU has a deterministic, single core streaming architecture that sets the standard for GenAI inference speed with predictable and repeatable performance for any given workload.

Beyond the architecture, our software is designed to empower developers like you with the tools you need to create innovative, powerful AI applications. With Groq as your engine, you can:

- Achieve uncompromised low latency and performance for real-time AI and HPC inferences 🔥
- Know the exact performance and compute time for any given workload 🔮
- Take advantage of our cutting-edge technology to stay ahead of the competition 💪

Want more Groq? Check out our [website](https://groq.com) for more resources and join our [Discord community](https://discord.gg/JvNsBDKeCG) to connect with our developers!

## Develop

To create a development environment, install poetry then run:

```bash
poetry install --with dev
```

## Testing

To test the integration, first enter the poetry venv:

```bash
poetry shell
```

Then tests can be run with make

```bash
make test
```

### Integration tests

Integration tests will be skipped unless an API key is provided. API keys can be created ath the [groq console](https://console.groq.com/keys).
Once created, store the API key in an environment variable and run tests

```bash
export GROQ_API_KEY=<your key here>
make test
```

## Linting and Formatting

Linting and code formatting can be executed with make.

```bash
make format
make lint
```
