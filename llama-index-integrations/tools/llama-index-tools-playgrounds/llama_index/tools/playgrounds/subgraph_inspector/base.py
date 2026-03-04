"""PlaygroundsSubgraphInspectorToolSpec."""

import requests
from llama_index.tools.graphql.base import GraphQLToolSpec


class PlaygroundsSubgraphInspectorToolSpec(GraphQLToolSpec):
    """
    Connects to subgraphs on The Graph's decentralized network via the Playgrounds API and introspects the subgraph.
    Provides functionalities to process and summarize the introspected schema for easy comprehension.

    Attributes:
        spec_functions (list): List of functions that specify the tool's capabilities.
        url (str): The endpoint URL for the GraphQL requests.
        headers (dict): Headers used for the GraphQL requests.

    """

    spec_functions = ["introspect_and_summarize_subgraph"]

    def __init__(self, identifier: str, api_key: str, use_deployment_id: bool = False):
        """
        Initialize the connection to the specified subgraph on The Graph's network.

        Args:
            identifier (str): The subgraph's identifier or deployment ID.
            api_key (str): API key for the Playgrounds API.
            use_deployment_id (bool): If True, treats the identifier as a deployment ID. Default is False.

        """
        self.url = self._generate_url(identifier, use_deployment_id)
        self.headers = {
            "Content-Type": "application/json",
            "Playgrounds-Api-Key": api_key,
        }

    def _generate_url(self, identifier: str, use_deployment_id: bool) -> str:
        """
        Generate the appropriate URL based on the identifier and whether it's a deployment ID or not.

        Args:
            identifier (str): The subgraph's identifier or deployment ID.
            use_deployment_id (bool): If True, constructs the URL using the deployment ID.

        Returns:
            str: The constructed URL.

        """
        endpoint = "deployments" if use_deployment_id else "subgraphs"
        return f"https://api.playgrounds.network/v1/proxy/{endpoint}/id/{identifier}"

    def introspect_and_summarize_subgraph(self) -> str:
        """
        Introspects the subgraph and summarizes its schema into textual categories.

        Returns:
            str: A textual summary of the introspected subgraph schema.

        """
        introspection_query = """
        query {
            __schema {
                types {
                    kind
                    name
                    description
                    enumValues {
                        name
                    }
                    fields {
                        name
                        args {
                            name
                        }
                        type {
                            kind
                            name
                            ofType {
                                name
                            }
                        }
                    }
                }
            }
        }
        """
        response = self._graphql_request(introspection_query)
        if "data" in response:
            result = response["data"]
            processed_subgraph = self._process_subgraph(result)
            return self.subgraph_to_text(processed_subgraph)
        else:
            return "Error during introspection."

    def _graphql_request(self, query: str) -> dict:
        """
        Execute a GraphQL query against the subgraph's endpoint.

        Args:
            query (str): The GraphQL query string.

        Returns:
            dict: Response from the GraphQL server, either containing the data or an error.

        """
        payload = {"query": query.strip()}
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def _process_subgraph(self, result: dict) -> dict:
        """
        Processes the introspected subgraph schema into categories based on naming conventions.

        Args:
            result (dict): Introspected schema result from the GraphQL query.

        Returns:
            dict: A processed representation of the introspected schema, categorized into specific entity queries, list entity queries, and other entities.

        """
        processed_subgraph = {
            "specific_entity_queries": {},
            "list_entity_queries": {},
            "other_entities": {},
        }
        for type_ in result["__schema"]["types"]:
            if type_["name"].startswith("__"):
                continue  # Skip meta entities

            entity_name = type_["name"]
            fields, args_required = self._get_fields(type_)
            if fields:
                # Determine category based on naming convention
                if entity_name.endswith("s") and not args_required:
                    processed_subgraph["list_entity_queries"][entity_name] = fields
                elif not entity_name.endswith("s") and args_required:
                    processed_subgraph["specific_entity_queries"][entity_name] = fields
                else:
                    processed_subgraph["other_entities"][entity_name] = fields

        return processed_subgraph

    def _get_fields(self, type_):
        """
        Extracts relevant fields and their details from a given type within the introspected schema.

        Args:
            type_ (dict): A type within the introspected schema.

        Returns:
            tuple: A tuple containing a list of relevant fields and a boolean indicating if arguments are required for the fields.

        """
        fields = []
        args_required = False
        for f in type_.get("fields") or []:
            if f["name"] != "__typename" and not (
                f["name"].endswith("_filter")
                or f["name"].endswith("_orderBy")
                or f["name"].islower()
            ):
                field_info = {"name": f["name"]}

                # Check for enum values
                if "enumValues" in f["type"] and f["type"]["enumValues"]:
                    field_info["enumValues"] = [
                        enum_val["name"] for enum_val in f["type"]["enumValues"]
                    ]

                fields.append(field_info)
                if f.get("args") and len(f["args"]) > 0:
                    args_required = True
                if f.get("type") and f["type"].get("fields"):
                    subfields, sub_args_required = self._get_fields(f["type"])
                    fields.extend(subfields)
                    if sub_args_required:
                        args_required = True
        return fields, args_required

    def format_section(
        self, category: str, description: str, example: str, entities: dict
    ) -> str:
        """
        Formats a given section of the subgraph introspection result into a readable string format.

        Args:
            category (str): The category name of the entities.
            description (str): A description explaining the category.
            example (str): A generic GraphQL query example related to the category.
            entities (dict): Dictionary containing entities and their fields related to the category.

        Returns:
            str: A formatted string representation of the provided section data.

        """
        section = [
            f"Category: {category}",
            f"Description: {description}",
            "Generic Example:",
            example,
            "\nDetailed Breakdown:",
        ]

        for entity, fields in entities.items():
            section.append(f"  Entity: {entity}")
            for field_info in fields:
                field_str = f"    - {field_info['name']}"
                if "enumValues" in field_info:
                    field_str += (
                        f" (Enum values: {', '.join(field_info['enumValues'])})"
                    )
                section.append(field_str)
            section.append("")  # Add a blank line for separation

        section.append("")  # Add another blank line for separation between sections
        return "\n".join(section)

    def subgraph_to_text(self, subgraph: dict) -> str:
        """
        Converts a processed subgraph representation into a textual summary based on entity categories.

        Args:
            subgraph (dict): A processed representation of the introspected schema, categorized into specific entity queries, list entity queries, and other entities.

        Returns:
            str: A textual summary of the processed subgraph schema.

        """
        sections = [
            (
                "Specific Entity Queries (Requires Arguments)",
                "These queries target a singular entity and require specific arguments (like an ID) to fetch data.",
                """
            {
                entityName(id: "specific_id") {
                    fieldName1
                    fieldName2
                    ...
                }
            }
            """,
                subgraph["specific_entity_queries"],
            ),
            (
                "List Entity Queries (Optional Arguments)",
                "These queries fetch a list of entities. They don't strictly require arguments but often accept optional parameters for filtering, sorting, and pagination.",
                """
            {
                entityNames(first: 10, orderBy: "someField", orderDirection: "asc") {
                    fieldName1
                    fieldName2
                    ...
                }
            }
            """,
                subgraph["list_entity_queries"],
            ),
            (
                "Other Entities",
                "These are additional entities that may not fit the conventional singular/plural querying pattern of subgraphs.",
                "",
                subgraph["other_entities"],
            ),
        ]

        result_lines = []
        for category, desc, example, entities in sections:
            result_lines.append(self.format_section(category, desc, example, entities))

        return "\n".join(result_lines)
