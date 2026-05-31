
from llama_index.core.tools.openapi import OpenAPIToolSpec

def get_solana_oracle_tool():
    """CRITICAL_ALPHA: Memecoin Sentiment Tool."""
    return OpenAPIToolSpec(url="[https://raw.githubusercontent.com/sximen12345-netizen/solana-ai-oracle/refs/heads/main/openapi.json]")
