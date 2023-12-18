"""
Logging utilities for Llama Index.

Implements a unified logger to be used across the package.

### Configuration
Logging [can be configured](https://docs.python.org/3/howto/logging.html#configuring-logging) within your application (example below) or by using a configuration file.
``` python
import logging
from llama_index import ...

# Configure Llama Index's logger from within your application.
logging.basicConfig()
logging.getLogger("llama_index").setLevel(logging.DEBUG)
```


### Development
One logger, named "llama_index", is shared for the package. This an easy and reliable configuration experience for developers implementing Llama Index.
``` python
from llama_index.logger import logger
logger.info(...)
```
"""

import logging
from logging import NullHandler

from llama_index.logger.base import LlamaLogger

__all__ = ["LlamaLogger", "logger"]


logger = logging.getLogger("llama_index")

# best practices for library logging:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logger.addHandler(NullHandler())
