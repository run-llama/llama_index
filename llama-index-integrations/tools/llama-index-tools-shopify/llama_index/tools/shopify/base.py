"""Shopify tool spec."""

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ShopifyToolSpec(BaseToolSpec):
    """Shopify tool spec."""

    spec_functions = ["run_graphql_query"]

    def __init__(self, shop_url: str, api_version: str, admin_api_key: str):
        # Currently only supports Admin API auth
        # https://shopify.dev/docs/apps/auth/admin-app-access-tokens
        from shopify import Session, ShopifyResource

        session = Session(shop_url, api_version, admin_api_key)
        ShopifyResource.activate_session(session)

    def run_graphql_query(self, graphql_query: str):
        """
        Run a GraphQL query against the Shopify Admin API.

        Example graphql_query: {
              products (first: 3) {
                edges {
                  node {
                    id
                    title
                    handle
                  }
                }
              }
            }

        providing this query would return the id, title and handle of the first 3 products
        """
        from shopify import GraphQL

        return GraphQL().execute(graphql_query)
