try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        PrivateAttr,
        root_validator,
        validator,
        create_model,
        StrictFloat,
        StrictInt,
        StrictStr,
    )
    from pydantic.v1.fields import FieldInfo
    from pydantic.v1.error_wrappers import ValidationError
except ImportError:
    from pydantic import (
        BaseModel,
        Field,
        PrivateAttr,
        root_validator,
        validator,
        create_model,
        StrictFloat,
        StrictInt,
        StrictStr,
    )
    from pydantic.fields import FieldInfo
    from pydantic.error_wrappers import ValidationError

__all__ = [
    "BaseModel",
    "Field",
    "PrivateAttr",
    "root_validator",
    "validator",
    "create_model",
    "StrictFloat",
    "StrictInt",
    "StrictStr",
    "FieldInfo",
    "ValidationError",
]
