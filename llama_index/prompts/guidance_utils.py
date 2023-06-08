from typing import Type

from pydantic import BaseModel


def convert_to_handlebars(text: str):
    """Convert a python format string to handlebars template.

    In python format string, single braces {} are used for variable substitution,
        and double braces {{}} are used for escaping actual braces (e.g. for JSON dict)
    In handlebars template, double braces {{}} are used for variable substitution,
        and single braces are actual braces (e.g. for JSON dict)
    """
    # Replace double braces with a temporary placeholder
    var_left = "TEMP_BRACE_LEFT"
    var_right = "TEMP_BRACE_RIGHT"
    text = text.replace("{{", var_left)
    text = text.replace("}}", var_right)

    # Replace single braces with double braces
    text = text.replace("{", "{{")
    text = text.replace("}", "}}")

    # Replace the temporary placeholder with single braces
    text = text.replace(var_left, "{")
    text = text.replace(var_right, "}")
    return text


def pydantic_to_guidance(cls: Type[BaseModel]) -> str:
    return json_schema_to_guidance(cls.schema())


def json_schema_to_guidance(
    schema: dict,
    key=None, 
    indent=0,
) -> str:
    out = ""
    if schema["type"] == "object":
        out += "  " * indent + "{\n"
        for k, v in schema["properties"].items():
            out += (
                "  " * (indent + 1)
                + k
                + ": "
                + json_schema_to_guidance(v, k, indent + 1)
                + ",\n"
            )
        out += "  " * indent + "}"
        return out
    elif schema["type"] == "array":
        if "max_items" in schema:
            extra_args = f" max_iterations={schema['max_items']}"
        else:
            extra_args = ""
        return (
            "[{{#geneach '"
            + key
            + "' stop=']'"
            + extra_args
            + "}}{{#unless @first}}, {{/unless}}"
            + json_schema_to_guidance(schema["items"], "this")
            + "{{/geneach}}]"
        )
    elif schema["type"] == "string":
        return "\"{{gen '" + key + "' stop='\"'}}\""
    elif schema["type"] == "number":
        return "{{gen '" + key + "' pattern='[0-9\\.]' stop=','}}"
    elif schema["type"] == "boolean":
        return "{{#select '" + key + "'}}True{{or}}False{{/select}}"
