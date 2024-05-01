"""Self Reflection Agent Worker."""

import logging

from llama_index.core.agent.types import (
    BaseAgentWorker,
)
from llama_index.core.bridge.pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SelfReflectionAgentWorker(BaseModel, BaseAgentWorker):
    ...
