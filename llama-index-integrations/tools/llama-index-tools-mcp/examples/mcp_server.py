import argparse
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, IPvAnyAddress
import ipinfo
import os
import dotenv

dotenv.load_dotenv()

# Create MCP server
mcp = FastMCP("BasicServer")


class IPDetails(BaseModel):
    ip: IPvAnyAddress = None  # type: ignore
    hostname: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    loc: str | None = None
    timezone: str | None = None

    class Config:
        extra = "ignore"


@mcp.tool()
async def fetch_ipinfo(ip: str | None = None, **kwargs) -> IPDetails:
    """Get the detailed information of a specified IP address

    Args:
        ip(str or None): The IP address to get information for. Follow the format like 192.168.1.1 .
        **kwargs: Additional keyword arguments to pass to the IPInfo handler.
    Returns:
        IPDetails: The detailed information of the specified IP address.
    """
    handler = ipinfo.getHandler(
        access_token=os.environ.get("IPINFO_API_TOKEN", None),
        headers={"user-agent": "basic-mcp-server", "custom_header": "yes"},
        **kwargs,
    )

    details = handler.getDetails(ip_address=ip)

    return IPDetails(**details.all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )
    args = parser.parse_args()
    mcp.run(args.server_type)
