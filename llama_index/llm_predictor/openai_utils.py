GPT4_MODELS = {
    # stable model names:
    #   resolves to gpt-4-0314 before 2023-06-27,
    #   resolves to gpt-4-0613 after
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    # 0314 models
    "gpt-4-0314": 8192,
    "gpt-4-32k-0314": 32768,
}

TURBO_MODELS = {
    # stable model names:
    #   resolves to gpt-3.5-turbo-0301 before 2023-06-27,
    #   resolves to gpt-3.5-turbo-0613 after
    "gpt-3.5-turbo": 4096,
    # resolves to gpt-3.5-turbo-16k-0613
    "gpt-3.5-turbo-16k": 16384,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    # 0301 models
    "gpt-3.5-turbo-0301": 4096,
}

GPT3_5_MODELS = {
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
}

GPT3_MODELS = {
    "text-ada-001": 2049,
    "text-babbage-001": 2040,
    "text-curie-001": 2049,
    "ada": 2049,
    "babbage": 2049,
    "curie": 2049,
    "davinci": 2049,
}

ALL_AVAILABLE_MODELS = {
    **GPT4_MODELS,
    **TURBO_MODELS,
    **GPT3_5_MODELS,
    **GPT3_MODELS,
}

DISCONTINUED_MODELS = {
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}


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
    # handling finetuned models
    if "ft-" in modelname:
        modelname = modelname.split(":")[0]

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"OpenAI model {modelname} has been discontinued. Please choose another model."
        )

    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size
