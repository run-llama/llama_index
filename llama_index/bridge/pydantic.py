try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

__all__ = ["BaseModel"]
