from typing import TYPE_CHECKING, Optional, Type, TypeVar

from llama_index.output_parsers.base import OutputParserException
from llama_index.output_parsers.utils import parse_json_markdown

if TYPE_CHECKING:
    from guidance import Program

from llama_index.bridge.pydantic import BaseModel


def convert_to_handlebars(text: str) -> str:
    """Convert a python format string to handlebars-style template.

    In python format string, single braces {} are used for variable substitution,
        and double braces {{}} are used for escaping actual braces (e.g. for JSON dict)
    In handlebars template, double braces {{}} are used for variable substitution,
        and single braces are actual braces (e.g. for JSON dict)

    This is currently only used to convert a python format string based prompt template
    to a guidance program template.
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
    return text.replace(var_right, "}")


def wrap_json_markdown(text: str) -> str:
    """Wrap text in json markdown formatting block."""
    return "```json\n" + text + "\n```"


def pydantic_to_guidance_output_template(cls: Type[BaseModel]) -> str:
    """Convert a pydantic model to guidance output template."""
    return json_schema_to_guidance_output_template(cls.schema(), root=cls.schema())


def pydantic_to_guidance_output_template_markdown(cls: Type[BaseModel]) -> str:
    """Convert a pydantic model to guidance output template wrapped in json markdown."""
    output = json_schema_to_guidance_output_template(cls.schema(), root=cls.schema())
    return wrap_json_markdown(output)


def json_schema_to_guidance_output_template(
    schema: dict,
    key: Optional[str] = None,
    indent: int = 0,
    root: Optional[dict] = None,
    use_pattern_control: bool = False,
) -> str:
    """Convert a json schema to guidance output template.

    Implementation based on https://github.com/microsoft/guidance/\
        blob/main/notebooks/applications/jsonformer.ipynb
    Modified to support nested pydantic models.
    """
    out = ""
    if "type" not in schema and "$ref" in schema:
        if root is None:
            raise ValueError("Must specify root schema for nested object")

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
        if key is None:
            raise ValueError("Key should not be None")
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
        if key is None:
            raise ValueError("key should not be None")
        return "\"{{gen '" + key + "' stop='\"'}}\""
    elif schema["type"] in ["integer", "number"]:
        if key is None:
            raise ValueError("key should not be None")
        if use_pattern_control:
            return "{{gen '" + key + "' pattern='[0-9\\.]' stop=','}}"
        else:
            return "\"{{gen '" + key + "' stop='\"'}}\""
    elif schema["type"] == "boolean":
        if key is None:
            raise ValueError("key should not be None")
        return "{{#select '" + key + "'}}True{{or}}False{{/select}}"
    else:
        schema_type = schema["type"]
        raise ValueError(f"Unknown schema type {schema_type}")


Model = TypeVar("Model", bound=BaseModel)


def parse_pydantic_from_guidance_program(
    program: "Program", cls: Type[Model], verbose: bool = False
) -> Model:
    """Parse output from guidance program.

    This is a temporary solution for parsing a pydantic object out of an executed
    guidance program.

    NOTE: right now we assume the output is the last markdown formatted json block

    NOTE: a better way is to extract via Program.variables, but guidance does not
          support extracting nested objects right now.
          So we call back to manually parsing the final text after program execution
    """
    try:
        output = program.text.split("```json")[-1]
        output = "```json" + output
        if verbose:
            print("Raw output:")
            print(output)
        json_dict = parse_json_markdown(output)
        sub_questions = cls.parse_obj(json_dict)
    except Exception as e:
        raise OutputParserException(
            "Failed to parse pydantic object from guidance program"
        ) from e
    return sub_questions
