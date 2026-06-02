MINIMAX_MODEL_TO_CONTEXT_WINDOW = {
    "MiniMax-M3": 524288,
    "MiniMax-M2.7": 204800,
    "MiniMax-M2.7-highspeed": 204800,
}

FUNCTION_CALLING_MODELS = {
    "MiniMax-M3",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
}


def get_context_window(model: str) -> int:
    return MINIMAX_MODEL_TO_CONTEXT_WINDOW.get(model, 204800)
