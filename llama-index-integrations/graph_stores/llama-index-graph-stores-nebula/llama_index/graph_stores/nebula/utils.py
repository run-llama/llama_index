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
