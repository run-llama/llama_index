import datetime
from typing import Dict, Any, Optional, Tuple
from nebula3.data.ResultSet import ResultSet
from nebula3.common.ttypes import Value, NList, Date, Time, DateTime


def url_scheme_parse(url) -> Tuple[str, int]:
    """
    Parse the URL scheme and host from the URL.

    Parameters:
    url (str): The URL to parse. i.e. nebula://localhost:9669

    Returns:
    Tuple[str, int]: A tuple containing the address and port.
    """
    scheme, address = url.split("://")
    if scheme not in ["nebula", "nebula3"]:
        raise ValueError(f"Invalid scheme {scheme}. Expected 'nebula' or 'nebula3'.")
    host, port = address.split(":")
    if not host:
        raise ValueError("Invalid host. Expected a hostname or IP address.")
    if not port:
        raise ValueError("Invalid port. Expected a port number.")
    if not port.isdigit():
        raise ValueError("Invalid port. Expected a number.")
    return host, int(port)


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns:
    dict: A new dictionary with all empty values removed.
    """
    # Create a new dictionary excluding empty values
    return {key: value for key, value in input_dict.items() if value}


def build_param_map(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a parameter map with proper binary values.
    """

    byte_params = {}
    for k, v in params.items():
        if isinstance(v, Value):
            byte_params[k] = v
        elif str(type(v)).startswith("nebula3.common.ttypes"):
            byte_params[k] = v
        else:
            byte_params[k] = _cast_value(v)
    return byte_params


def _cast_value(value: Any) -> Value:
    """
    Cast the value to nebula Value type
    ref: https://github.com/vesoft-inc/nebula/blob/master/src/common/datatypes/Value.cpp
    :param value: the value to be casted
    :return: the casted value
    """
    casted_value = Value()
    if isinstance(value, bool):
        casted_value.set_bVal(value)
    elif isinstance(value, int):
        casted_value.set_iVal(value)
    elif isinstance(value, str):
        casted_value.set_sVal(value)
    elif isinstance(value, float):
        casted_value.set_fVal(value)
    elif isinstance(value, datetime.date):
        date_value = Date(year=value.year, month=value.month, day=value.day)
        casted_value.set_dVal(date_value)
    elif isinstance(value, datetime.time):
        time_value = Time(
            hour=value.hour,
            minute=value.minute,
            sec=value.second,
            microsec=value.microsecond,
        )
        casted_value.set_tVal(time_value)
    elif isinstance(value, datetime.datetime):
        datetime_value = DateTime(
            year=value.year,
            month=value.month,
            day=value.day,
            hour=value.hour,
            minute=value.minute,
            sec=value.second,
            microsec=value.microsecond,
        )
        casted_value.set_dtVal(datetime_value)
    # TODO: add support for GeoSpatial
    elif isinstance(value, list):
        byte_list = []
        for item in value:
            byte_list.append(_cast_value(item))
        casted_value.set_lVal(NList(values=byte_list))
    elif isinstance(value, dict):
        # TODO: add support for NMap
        raise TypeError("Unsupported type: dict")
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
    return casted_value

def deduce_property_types_from_values(property_values: Dict[str, Any]) -> Dict[str, str]:
    """
    Deduce the data types of properties for NebulaGraph DDL based on the property values.

    Parameters:
    property_values (dict): The properties of the tag.

    Returns:
    dict: A dictionary mapping property names to their deduced data types.
    """
    property_type_mapping = {}
    for property_name, value in property_values.items():
        if isinstance(value, bool):
            property_type_mapping[property_name] = "bool"
        elif isinstance(value, int):
            property_type_mapping[property_name] = "int"
        elif isinstance(value, float):
            property_type_mapping[property_name] = "double"
        elif isinstance(value, str):
            property_type_mapping[property_name] = "string"
        else:
            raise ValueError(f"Unsupported property type: {type(value)}")
    return property_type_mapping

def generate_ddl_create_tag(tag_name: str, properties: Dict[str, Any]) -> str:
    """
    Generate the DDL to create a NebulaGraph tag.

    Parameters:
    tag_name (str): The name of the tag.
    properties (dict): The properties of the tag.

    Returns:
    str: The DDL string.
    """
    # infer properties type in NebulaGraph DDL
    property_type_map = deduce_property_types_from_values(properties)

    # generate DDL
    ddl_parts = [f"CREATE TAG `{tag_name}` ("]
    prop_definitions = []
    for prop, dtype in property_type_map.items():
        prop_definition = f"`{prop}` {dtype} NULL"
        prop_definitions.append(prop_definition)
    ddl_parts.append(", ".join(prop_definitions))
    ddl_parts.append(");")
    ddl_statement = " ".join(ddl_parts)
    return ddl_statement

def generate_ddl_alter_tag(tag_name: str, existing_property_type_map: Dict[str, str], new_properties: Dict[str, Any], perform_prop_drop_if_missing: bool = False) -> Optional[str]:
    """
    Generate the DDL to alter a NebulaGraph tag.

    Parameters:
    tag_name (str): The name of the tag.
    existing_property_type_map (dict): The existing properties of the tag.
    new_properties (dict): The new properties to add to the tag.
    perform_prop_drop_if_missing (bool): Whether to drop properties that are not in the new properties.

    Returns:
    str: The DDL string.
    """

    # infer properties type in NebulaGraph DDL
    new_property_type_map = deduce_property_types_from_values(new_properties)

    # generate DDL
    ddl_parts = [f"ALTER TAG `{tag_name}` ADD ("]
    prop_definitions = []
    for prop, dtype in new_property_type_map.items():
        if prop not in existing_property_type_map:
            prop_definition = f"`{prop}` {dtype} NULL"
            prop_definitions.append(prop_definition)
        elif existing_property_type_map[prop] != dtype:
            raise ValueError(f"Property {prop} already exists with a different type {existing_property_type_map[prop]}")

    if prop_definitions:
        ddl_parts.append(", ".join(prop_definitions))
        ddl_parts.append(");")
        ddl_statement = " ".join(ddl_parts)
    else:
        ddl_statement = None

    if perform_prop_drop_if_missing:
        ddl_parts = [f"ALTER TAG `{tag_name}` DROP ("]
        prop_definitions = []
        for prop, dtype in new_property_type_map.items():
            if prop in existing_property_type_map:
                prop_definition = f"`{prop}`"
                prop_definitions.append(prop_definition)
        if prop_definitions:
            ddl_parts.append(", ".join(prop_definitions))
            ddl_parts.append(");")
            if ddl_statement is None:
                ddl_statement = ""
            ddl_statement += " ".join(ddl_parts)

    return ddl_statement
