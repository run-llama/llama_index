import json
from typing import Optional, Type, Any

from llama_index.core.tools import ToolMetadata
from llama_index.core.tools.types import DefaultToolFnSchema
from pydantic import BaseModel


def remap_schema(schema: dict) -> dict:
    """
    Remap schema to match Gemini's internal API.
    """
    parameters = {}

    for key, value in schema.items():
        if key in ["title", "type", "properties", "required", "definitions"]:
            parameters[key] = value
        elif key == "$ref":
            parameters["defs"] = value
        else:
            continue

    return parameters


class GeminiToolMetadataWrapper:
    """
    The purpose of this dataclass is to represent the metadata in
    a manner that is compatible with Gemini's internal APIs. The
    default ToolMetadata class generates a json schema using $ref
    and $def field types which break google's protocol buffer
    serialization.
    """

    def __init__(self, base: ToolMetadata) -> None:
        self._base = base
        self._name = self._base.name
        self._description = self._base.description
        self._fn_schema = self._base.fn_schema
        self._parameters = self.get_parameters_dict()

    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema

    def get_parameters_dict(self) -> dict:
        parameters = {}

        if self.fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = remap_schema(
                {
                    k: v
                    for k, v in self.fn_schema.model_json_schema()
                    if k in ["type", "properties", "required", "definitions", "$defs"]
                }
            )

        return parameters

    @property
    def fn_schema_str(self) -> str:
        """Get fn schema as string."""
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        parameters = self.get_parameters_dict()
        return json.dumps(parameters)

    def __getattr__(self, item) -> Any:
        match item:
            case "name":
                return self._name
            case "description":
                return self._description
            case "fn_schema":
                return self.fn_schema
            case "parameters":
                return self._parameters
            case _:
                raise AttributeError(
                    f"No attribute '{item}' found in GeminiToolMetadataWrapper"
                )


class GeminiToolWrapper:
    """
    Wraps a base tool object to make it compatible with Gemini's
    internal APIs.
    """

    def __init__(self, base_obj, *args, **kwargs) -> None:
        self.base_obj = base_obj
        # some stuff

    @property
    def metadata(self) -> GeminiToolMetadataWrapper:
        return GeminiToolMetadataWrapper(self.base_obj.metadata)

    def __getattr__(self, name) -> Any:
        return getattr(self.base_obj, name)
