import re

MISTRALAI_CODE_MODELS = "codestral-latest"

THINKING_REGEX = re.compile(r"^<think>\n(.*?)\n</think>\n")
THINKING_START_REGEX = re.compile(r"^<think>\n")
