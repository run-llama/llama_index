from typing import Optional, Type

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


def wrap_json_markdown(text: str) -> str:
    return "```json\n" + text + "\n```"


def pydantic_to_guidance_output_template(cls: Type[BaseModel]) -> str:
    output = json_schema_to_guidance_output_template(cls.schema(), root=cls.schema())
    return wrap_json_markdown(output)


def json_schema_to_guidance_output_template(
    schema: dict,
    key: Optional[str] = None,
    indent: int = 0,
    root: Optional[dict] = None,
) -> str:
    out = ""
    if "type" not in schema and "$ref" in schema:
        ref = schema["$ref"]
        model = ref.split("/")[-1]
        return json_schema_to_guidance_output_template(
            root["definitions"][model], key, indent, root
        )

    if schema["type"] == "object":
        out += "  " * indent + "{\n"
        for k, v in schema["properties"].items():
            out += (
                "  " * (indent + 1)
                + f'"{k}"'
                + ": "
                + json_schema_to_guidance_output_template(v, k, indent + 1, root)
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
            + json_schema_to_guidance_output_template(schema["items"], "this", 0, root)
            + "{{/geneach}}]"
        )
    elif schema["type"] == "string":
        return "\"{{gen '" + key + "' stop='\"'}}\""
    elif schema["type"] == "number":
        return "{{gen '" + key + "' pattern='[0-9\\.]' stop=','}}"
    elif schema["type"] == "boolean":
        return "{{#select '" + key + "'}}True{{or}}False{{/select}}"
