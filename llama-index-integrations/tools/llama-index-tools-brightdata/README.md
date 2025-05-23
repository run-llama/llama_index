# LlamaIndex Tools Integration: Bright Data

This tool connects to [Bright Data](https://brightdata.com/) to enable your agent to crawl websites, search the web, and access structured data from platforms like LinkedIn, Amazon, and social media.

Bright Data's tools provide robust web scraping capabilities with built-in CAPTCHA solving and bot detection avoidance, allowing you to reliably extract data from the web.

## Installation

```bash
pip install llama-index llama-index-core llama-index-tools-brightdata
```

## Authentication

To use this tool, you'll need a Bright Data API key. You can obtain one by signing up for a Bright Data account on their [website](https://brightdata.com/).

## Usage

Here's an example of how to use the BrightDataToolSpec with LlamaIndex:

```python
llm = OpenAI(model="gpt-4o", api_key="your-api-key")

brightdata_tool = BrightDataToolSpec(api_key="your-api-key", zone="unlocker")

tool_list = brightdata_tool.to_tool_list()

for tool in tool_list:
    tool.original_description = tool.metadata.description
    tool.metadata.description = "Bright Data web scraping tool"

agent = OpenAIAgent.from_tools(tools=tool_list, llm=llm)

query = (
    "Find and summarize the latest news about AI from major tech news sites"
)
tool_descriptions = "\n\n".join(
    [
        f"Tool Name: {tool.metadata.name}\nTool Description: {tool.original_description}"
        for tool in tool_list
    ]
)

query_with_descriptions = f"{tool_descriptions}\n\nQuery: {query}"

response = agent.chat(query_with_descriptions)
print(response)
```

## Features

The Bright Data tool provides the following capabilities:

### Web Scraping

- `scrape_as_markdown`: Scrape a webpage and convert the content to Markdown format. This tool can bypass CAPTCHA and bot detection.

```python
result = brightdata_tool.scrape_as_markdown("https://example.com")
print(result.text)
```

### Visual Capture

- `get_screenshot`: Take a screenshot of a webpage and save it to a file.

```python
screenshot_path = brightdata_tool.get_screenshot(
    "https://example.com", output_path="example_screenshot.png"
)
```

### Search Engine Access

- `search_engine`: Search Google, Bing, or Yandex and get structured search results as JSON or Markdown. Supports advanced parameters for more specific searches.

```python
search_results = brightdata_tool.search_engine(
    query="climate change solutions",
    engine="google",
    language="en",
    country_code="us",
    num_results=20,
)
print(search_results.text)
```

### Structured Web Data Extraction

- `web_data_feed`: Retrieve structured data from various platforms including LinkedIn, Amazon, Instagram, Facebook, X (Twitter), Zillow, and more.

```python
linkedin_profile = brightdata_tool.web_data_feed(
    source_type="linkedin_person_profile",
    url="https://www.linkedin.com/in/username/",
)
print(linkedin_profile)

amazon_product = brightdata_tool.web_data_feed(
    source_type="amazon_product", url="https://www.amazon.com/dp/B08N5KWB9H"
)
print(amazon_product)
```

## Advanced Configuration

The Bright Data tool offers various configuration options for specialized use cases:

### Search Engine Parameters

The `search_engine` function supports advanced parameters like:

- Language targeting (`language` parameter)
- Country-specific search (`country_code` parameter)
- Different search types (images, shopping, news, etc.)
- Pagination controls
- Mobile device emulation
- Geolocation targeting
- Hotel search parameters

```python
results = brightdata_tool.search_engine(
    query="best hotels in paris",
    engine="google",
    language="fr",
    country_code="fr",
    search_type="shopping",
    device="mobile",
    hotel_dates="2025-06-01,2025-06-05",
    hotel_occupancy=2,
)
```

### Supported Web Data Sources

The `web_data_feed` function supports retrieving structured data from:

- LinkedIn (profiles and companies)
- Amazon (products and reviews)
- Instagram (profiles, posts, reels, comments)
- Facebook (posts, marketplace listings, company reviews)
- X/Twitter (posts)
- Zillow (property listings)
- Booking.com (hotel listings)
- YouTube (videos)
- ZoomInfo (company profiles)

For more information, visit the [Bright Data documentation](https://docs.brightdata.com/).
