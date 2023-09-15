try:
    import pydantic.v1 as pydantic
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
    from pydantic.v1.generics import GenericModel
except ImportError:
    import pydantic
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
    from pydantic.generics import GenericModel

__all__ = [
    "pydantic",
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
    "GenericModel",
]
