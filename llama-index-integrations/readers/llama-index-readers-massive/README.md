# Massive Web Reader

Loads web pages using [Massive](https://joinmassive.com) proxy network with Playwright browser automation. Massive provides access to a global network of residential proxies with comprehensive geotargeting capabilities.

## Features

- **Country targeting**: 195+ countries with automatic locale and timezone configuration
- **City and ZIP code geotargeting**: target specific geographic locations
- **ASN targeting**: target specific Autonomous System Numbers
- **Device type targeting**: mobile, tv or common (any non-mobile) device
- **Sticky sessions**: maintain the same IP across multiple requests with configurable TTL
- **Raw HTML mode**: option to return unprocessed HTML

## Installation

```bash
pip install llama-index-readers-massive
playwright install chromium
```

or

```bash
uv add llama-index-readers-massive
playwright install chromium
```

## Usage

### Get Massive Residential Proxy credentials

[Sign up](https://www.joinmassive.com/residential-proxies) to get the username and password.

### Basic Usage

```python
from llama_index.readers.massive import MassiveWebReader

# Initialize with Massive proxy credentials
reader = MassiveWebReader(
    username="your_massive_username",
    password="your_massive_password",
    country="US"
)

# Load documents from URLs
documents = reader.load_data(["https://example.com"])
```

### City Targeting with Mobile Device

```python
reader = MassiveWebReader(
    username="your_massive_username",
    password="your_massive_password",
    country="US",
    city="New York",
    device_type="mobile"
)
```

### Sticky Session

```python
reader = MassiveWebReader(
    username="your_massive_username",
    password="your_massive_password",
    country="GB",
    session="my-session-123",
    ttl=30  # 30 minutes
)
```

### Raw HTML Mode

```python
reader = MassiveWebReader(
    username="your_massive_username",
    password="your_massive_password",
    country="DE",
    raw_html=True
)
```

### Async Loading

```python
import asyncio
from llama_index.readers.massive import MassiveWebReader

async def main():
    reader = MassiveWebReader(
        username="your_massive_username",
        password="your_massive_password",
        country="US"
    )
    documents = await reader.aload_data(["https://example.com"])
    return documents

documents = asyncio.run(main())
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `username` | str | None | Massive proxy username |
| `password` | str | None | Massive proxy password |
| `country` | str | None | ISO 3166-1 alpha-2 country code (e.g., 'US', 'DE', 'GB') |
| `city` | str | None | City name for geotargeting (e.g., 'New York', 'London') |
| `zipcode` | str | None | ZIP/postal code (e.g., '10001', 'SW1') |
| `asn` | str | None | ASN identifier for network targeting |
| `device_type` | str | None | Device type: 'mobile', 'common', or 'tv' |
| `session` | str | None | Session identifier for sticky sessions |
| `ttl` | int | 15 | Session TTL in minutes |
| `headless` | bool | True | Run browser in headless mode |
| `page_load_timeout` | int | 30000 | Maximum time in ms to wait for page load |
| `additional_wait_ms` | int | None | Extra wait time after networkidle for lazy-loaded content |
| `raw_html` | bool | False | Return raw HTML without BeautifulSoup processing |

## Requirements

- Python >= 3.10
- Playwright (with Chromium browser)
- BeautifulSoup4
- A [Massive](https://joinmassive.com) account with proxy credentials
