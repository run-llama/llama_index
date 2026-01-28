# LlamaIndex Tools: APIVerve

Access 300+ utility APIs for AI agents through LlamaIndex.

[APIVerve](https://apiverve.com) provides a comprehensive collection of utility APIs including validation, conversion, generation, analysis, and lookup tools.

## Installation

```bash
pip install llama-index-tools-apiverve
```

## Usage

```python
from llama_index.tools.apiverve import APIVerveToolSpec
from llama_index.agent.openai import OpenAIAgent

# Initialize with your API key
apiverve = APIVerveToolSpec(api_key="your-api-key")
# Or set APIVERVE_API_KEY environment variable

# Create an agent with APIVerve tools
agent = OpenAIAgent.from_tools(apiverve.to_tool_list())

# Use the agent
response = agent.chat("Is test@example.com a valid email address?")
print(response)
```

## Available Tools

### call_api

Call any of the 300+ APIVerve APIs by ID.

```python
# Email validation
result = apiverve.call_api("emailvalidator", {"email": "test@example.com"})

# DNS lookup
result = apiverve.call_api("dnslookup", {"domain": "example.com"})

# IP geolocation
result = apiverve.call_api("iplookup", {"ip": "8.8.8.8"})

# QR code generation
result = apiverve.call_api("qrcodegenerator", {"value": "https://example.com"})

# Currency conversion
result = apiverve.call_api("currencyconverter", {"from": "USD", "to": "EUR", "amount": 100})
```

### list_available_apis

Discover available APIs and their descriptions.

```python
# List all APIs
apis = apiverve.list_available_apis()

# Filter by category
validation_apis = apiverve.list_available_apis(category="Validation")

# Search by keyword
email_apis = apiverve.list_available_apis(search="email")
```

### list_categories

List all available API categories.

```python
categories = apiverve.list_categories()
# ['Analysis', 'Conversion', 'Generation', 'Lookup', 'Validation', ...]
```

## API Categories

APIVerve provides APIs across many categories:

| Category | Examples |
|----------|----------|
| **Validation** | Email, phone, URL, credit card validation |
| **Lookup** | DNS, IP, WHOIS, company info |
| **Generation** | QR codes, barcodes, passwords, UUIDs |
| **Conversion** | Currency, units, file formats |
| **Analysis** | Sentiment, readability, text extraction |
| **Utilities** | Hashing, encoding, randomization |

Browse all APIs at [apiverve.com/marketplace](https://apiverve.com/marketplace)

## Getting an API Key

1. Sign up at [dashboard.apiverve.com](https://dashboard.apiverve.com)
2. Create a new API key
3. Use it in the `api_key` parameter or set `APIVERVE_API_KEY` environment variable

## Resources

- [APIVerve Website](https://apiverve.com)
- [API Documentation](https://docs.apiverve.com)
- [API Marketplace](https://apiverve.com/marketplace)

## License

MIT License - see [LICENSE](LICENSE) for details.
