def openai_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = openai.modelname_to_contextsize("text-davinci-003")

    Modified from:
        https://github.com/hwchase17/langchain/blob/master/langchain/llms/openai.py
    """
    model_token_mapping = {
        "gpt-4-32k-0613": 32768,
        "gpt-4-0613": 8192,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-4": 8192,
        "gpt-4-0314": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0314": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0301": 4096,
        "text-ada-001": 2049,
        "ada": 2049,
        "text-babbage-001": 2040,
        "babbage": 2049,
        "text-curie-001": 2049,
        "curie": 2049,
        "davinci": 2049,
        "text-davinci-003": 4097,
        "text-davinci-002": 4097,
        "code-davinci-002": 8001,
        "code-davinci-001": 8001,
        "code-cushman-002": 2048,
        "code-cushman-001": 2048,
    }

    # handling finetuned models
    if "ft-" in modelname:
        modelname = modelname.split(":")[0]

    context_size = model_token_mapping.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(model_token_mapping.keys())
        )

    return context_size
