"""Init file for langchain helpers."""

try:
    import langchain  # noqa  # pants: no-infer-dep
except ImportError:
    raise ImportError(
        "langchain not installed. "
        "Please install langchain with `pip install llama_index[langchain]`."
    )
