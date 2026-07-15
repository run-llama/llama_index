MINIMAX_MODEL_TO_CONTEXT_WINDOW = {
    "MiniMax-M2.7": 204800,
    "MiniMax-M2.7-highspeed": 204800,
    "MiniMax-M2.5": 204800,
    "MiniMax-M2.5-highspeed": 204800,
}

FUNCTION_CALLING_MODELS = {
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
}


def get_context_window(model: str) -> int:
    return MINIMAX_MODEL_TO_CONTEXT_WINDOW.get(model, 204800)
