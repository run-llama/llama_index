# Model Context Protocol (MCP)

The Model Context Protocol (MCP) is an open-source standard protocol that allows Large Language Models (LLMs) to interact with external tools and data sources through structured API calls.

MCP serves as a standardization layer for AI applications to communicate effectively with external services such as tools, databases and predefined templates. Think of MCP as a "USB-C port" for AI applications - it provides a standardized way for various tools, platforms, and data sources to connect to AI models.

## Architecture

MCP operates through a client-server architecture:

- **MCP Hosts**: Applications like Claude Desktop, IDEs, or AI tools that wish to access data via MCP
- **MCP Clients**: Protocol clients that maintain 1:1 connections with MCP servers
- **MCP Servers**: Lightweight services that expose capabilities (tools, resources, prompts) via the standardized protocol

## Core Capabilities

MCP supports three main types of capabilities:

1. **Tools**: Functions that can be invoked with structured inputs
2. **Resources**: Data sources that can be read (files, databases, etc.)
3. **Prompts**: Reusable prompt templates with parameters

## With LlamaIndex

MCP introduces a compelling alternative to vector indexing for some use cases. Instead of crawling Salesforce data into a vector store for example, you can expose Salesforce as a live queryable tool.

This is particularly valuable for:

- **Real-time Data**: Get fresh data directly from source systems at query time
- **Structured Queries**: Handle complex queries requiring relationships and logic
- **Action-Taking**: Execute operations beyond just information retrieval

## Next Steps

- [Using MCP Tools with LlamaIndex](./llamaindex_mcp.md)
- [Use LlamaCloud APIs as MCP Tools/Servers](./llamacloud_mcp.md)
- [Use any existing LlamaIndex workflow/tool as an MCP Tool/Server](./convert_existing.md)
