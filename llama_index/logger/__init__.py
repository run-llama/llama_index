"""Init params."""

import logging
from logging import NullHandler

from llama_index.logger.base import LlamaLogger

__all__ = ["LlamaLogger"]

logger = logging.getLogger(__name__)

# best practices for library logging:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logger.addHandler(NullHandler())
