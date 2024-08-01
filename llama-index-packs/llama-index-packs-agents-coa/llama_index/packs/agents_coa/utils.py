import json


def json_schema_to_python(schema: str, name: str, description: str = None):
    """Helper function to convert a JSON schema to a Python function definition."""
    # Parse the JSON schema
    schema_data = json.loads(schema)

    # Generate function definition
    properties = schema_data.get("properties", {})
    required = schema_data.get("required", [])

    func_def = f"def {name}("
    for prop_name, prop_data in properties.items():
        ref = prop_data.get("$ref")
        if ref:
            type_name = ref.split("/")[-1]
            if prop_name in required:
                func_def += f"{prop_name}: {type_name}, "
            else:
                func_def += f"{prop_name}: Optional[{type_name}] = None, "
        else:
            prop_type = prop_data.get("type")
            if prop_name in required:
                func_def += f"{prop_name}: {prop_type}, "
            else:
                func_def += f"{prop_name}: Optional[{prop_type}] = None, "

    if description:
        func_def = func_def.rstrip(", ") + f'):\n   """{description}"""\n    ...'
    else:
        func_def = func_def.rstrip(", ") + "):\n    ..."

    return func_def
