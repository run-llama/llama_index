from typing import Union, Dict, Any, List
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dateutil import parser


class NeptuneQueryException(Exception):
    """Exception for the Neptune queries."""

    def __init__(self, exception: Union[str, Dict]):
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


def remove_empty_values(input_dict: Dict) -> Dict:
    """
    Remove entries with empty values from the dictionary.

    Parameters
    ----------
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns
    -------
    dict: A new dictionary with all empty values removed.

    """
    # Create a new dictionary excluding empty values
    output_dict = {key: value for key, value in input_dict.items() if value}
    return output_dict or {}


def create_neptune_database_client(
    host: str,
    port: int,
    provided_client: Any,
    credentials_profile_name: str,
    region_name: str,
    sign: bool,
    use_https: bool,
):
    """
    Create a Neptune Database Client.

    Args:
            host (str): The host endpoint
            port (int, optional): The port. Defaults to 8182.
            client (Any, optional): If provided, this is the client that will be used. Defaults to None.
            credentials_profile_name (Optional[str], optional): If provided this is the credentials profile that will be used. Defaults to None.
            region_name (Optional[str], optional): The region to use. Defaults to None.
            sign (bool, optional): True will SigV4 sign all requests, False will not. Defaults to True.
            use_https (bool, optional): True to use https, False to use http. Defaults to True.

    Returns:
        Any: The neptune client

    """
    try:
        client = None
        if provided_client is not None:
            client = provided_client
        else:
            if credentials_profile_name is not None:
                session = boto3.Session(profile_name=credentials_profile_name)
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {}
            if region_name:
                client_params["region_name"] = region_name

            protocol = "https" if use_https else "http"

            client_params["endpoint_url"] = f"{protocol}://{host}:{port}"

            if sign:
                client = session.client("neptunedata", **client_params)
            else:
                client = session.client(
                    "neptunedata",
                    **client_params,
                    config=Config(signature_version=UNSIGNED),
                )
        return client
    except ImportError:
        raise ModuleNotFoundError(
            "Could not import boto3 python package. "
            "Please install it with `pip install boto3`."
        )
    except Exception as e:
        if type(e).__name__ == "UnknownServiceError":
            raise ModuleNotFoundError(
                "Neptune Database requires a boto3 version 1.34.40 or greater."
                "Please install it with `pip install -U boto3`."
            ) from e
        else:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e


def create_neptune_analytics_client(
    graph_identifier: str,
    provided_client: Any,
    credentials_profile_name: str,
    region_name: str,
):
    """
    Creates a Neptune Analytics Client.

    Args:
        graph_identifier (str): The geaph identifier
        provided_client (Any): A client
        credentials_profile_name (str): The profile name
        region_name (str): The region name

    Returns:
        Any: The client

    """
    try:
        if not graph_identifier.startswith("g-") or "." in graph_identifier:
            raise ValueError(
                "The graph_identifier provided is not a valid value.  The graph identifier for a Neptune Analytics graph must be in the form g-XXXXXXXXXX."
            )

        client = None
        if provided_client is not None:
            client = client
        else:
            if credentials_profile_name is not None:
                session = boto3.Session(profile_name=credentials_profile_name)
            else:
                # use default credentials
                session = boto3.Session()

            if region_name:
                client = session.client(
                    "neptune-graph",
                    region_name=region_name,
                    config=(
                        Config(
                            retries={"total_max_attempts": 1, "mode": "standard"},
                            read_timeout=None,
                        )
                    ),
                )
            else:
                client = session.client(
                    "neptune-graph",
                    config=(
                        Config(
                            retries={"total_max_attempts": 1, "mode": "standard"},
                            read_timeout=None,
                        )
                    ),
                )
        return client
    except ImportError:
        raise ModuleNotFoundError(
            "Could not import boto3 python package. "
            "Please install it with `pip install boto3`."
        )
    except Exception as e:
        if type(e).__name__ == "UnknownServiceError":
            raise ModuleNotFoundError(
                "NeptuneGraph requires a boto3 version 1.34.40 or greater."
                "Please install it with `pip install -U boto3`."
            ) from e
        else:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e


def refresh_structured_schema(query: str, summary: Dict) -> None:
    """Refreshes the Neptune graph schema information."""


def refresh_schema(query: str, summary: Dict) -> None:
    """Refreshes the Neptune graph schema information."""
    types = {
        "str": "STRING",
        "float": "DOUBLE",
        "int": "INTEGER",
        "list": "LIST",
        "dict": "MAP",
        "bool": "BOOLEAN",
        "datetime": "DATETIME",
    }
    n_labels = summary["nodeLabels"]
    e_labels = summary["edgeLabels"]
    triple_schema = _get_triples(query, e_labels)
    node_properties = _get_node_properties(query, n_labels, types)
    edge_properties = _get_edge_properties(query, e_labels, types)
    structured_schema = {
        "node_labels": n_labels,
        "edge_labels": e_labels,
        "node_properties": node_properties,
        "edge_properties": edge_properties,
        "triples": triple_schema,
    }

    return {
        "schema_str": f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {triple_schema}
        """,
        "structured_schema": structured_schema,
    }


def _get_triples(query: str, e_labels: List[str]) -> List[str]:
    """Get the node-edge->node triple combinations."""
    triple_query = """
        MATCH (a)-[e:`{e_label}`]->(b)
        WITH a,e,b LIMIT 3000
        RETURN DISTINCT labels(a) AS from, type(e) AS edge, labels(b) AS to
        LIMIT 10
        """

    triple_template = "(:`{a}`)-[:`{e}`]->(:`{b}`)"
    triple_schema = []
    for label in e_labels:
        q = triple_query.format(e_label=label)
        data = query(q)
        for d in data:
            triple = triple_template.format(a=d["from"][0], e=d["edge"], b=d["to"][0])
            triple_schema.append(triple)

    return triple_schema


def _get_node_properties(query: str, n_labels: List[str], types: Dict) -> List:
    """Get the node properties for the label."""
    node_properties_query = """
        MATCH (a:`{n_label}`)
        RETURN properties(a) AS props
        LIMIT 100
        """
    node_properties = []
    for label in n_labels:
        q = node_properties_query.format(n_label=label)
        data = {"label": label, "properties": query(q)}
        s = set({})
        for p in data["properties"]:
            for k, v in p["props"].items():
                data_type = types[type(v).__name__]
                if types[type(v).__name__] == "STRING":
                    try:
                        if bool(parser.parse(v)):
                            data_type = "DATETIME"
                        else:
                            data_type = "STRING"
                    except (ValueError, OverflowError):
                        data_type = "STRING"
                s.add((k, data_type))

        np = {
            "properties": [{"property": k, "type": v} for k, v in s],
            "labels": label,
        }
        node_properties.append(np)

    return node_properties


def _get_edge_properties(
    query: str, e_labels: List[str], types: Dict[str, Any]
) -> List:
    """Get the edge properties for the label."""
    edge_properties_query = """
        MATCH ()-[e:`{e_label}`]->()
        RETURN properties(e) AS props
        LIMIT 100
        """
    edge_properties = []
    for label in e_labels:
        q = edge_properties_query.format(e_label=label)
        data = {"label": label, "properties": query(q)}
        s = set({})
        for p in data["properties"]:
            for k, v in p["props"].items():
                data_type = types[type(v).__name__]
                if types[type(v).__name__] == "STRING":
                    try:
                        if bool(parser.parse(v)):
                            data_type = "DATETIME"
                        else:
                            data_type = "STRING"
                    except ValueError:
                        data_type = "STRING"
                s.add((k, data_type))

        ep = {
            "type": label,
            "properties": [{"property": k, "type": v} for k, v in s],
        }
        edge_properties.append(ep)

    return edge_properties
