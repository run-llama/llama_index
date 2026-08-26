from typing import Dict, Set

# Model IDs below were verified against POST /api/v3/chat/completions on
# 2026-08-26. Volcengine Ark resolves fully versioned IDs only -- the short
# names shown in the console (e.g. "doubao-seed-2.1-pro") are display labels
# and return InvalidEndpointOrModel.NotFound. "doubao-seed-evolving" is the
# one exception: it is a rolling alias with no date suffix.
# Context windows: https://www.volcengine.com/docs/82379/1330310
VOLCENGINE_MODEL_TO_CONTEXT_WINDOW: Dict[str, int] = {
    # Doubao Seed 2.1
    "doubao-seed-2-1-pro-260628": 256000,
    "doubao-seed-2-1-turbo-260628": 256000,
    # Doubao Seed Evolving (rolling alias)
    "doubao-seed-evolving": 1024000,
    # Doubao Seed 2.0
    "doubao-seed-2-0-pro-260215": 256000,
    "doubao-seed-2-0-lite-260428": 256000,
    "doubao-seed-2-0-mini-260428": 256000,
    # DeepSeek on Ark
    "deepseek-v4-pro-260425": 1024000,
    "deepseek-v4-pro-ga-260813": 1024000,
    "deepseek-v4-flash-260425": 1024000,
    "deepseek-v4-flash-ga-260731": 1024000,
    # GLM on Ark
    "glm-5-2-260617": 1024000,
}

FUNCTION_CALLING_MODELS: Set[str] = set(VOLCENGINE_MODEL_TO_CONTEXT_WINDOW)

DEFAULT_CONTEXT_WINDOW = 256000


def get_context_window(model: str) -> int:
    """
    Return the context window for a model, falling back to a conservative
    default so newly released Ark models still work without a library bump.
    """
    return VOLCENGINE_MODEL_TO_CONTEXT_WINDOW.get(model, DEFAULT_CONTEXT_WINDOW)
