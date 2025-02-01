DEEPSEEK_MODEL_TO_CONTEXT_WINDOW = {
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
}

FUNCTION_CALLING_MODELS = {"deepseek-chat"}


def get_context_window(model: str) -> int:
    return DEEPSEEK_MODEL_TO_CONTEXT_WINDOW.get(model, 64000)
