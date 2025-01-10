from llama_index.tools.scrapegraph import ScrapegraphToolSpec

# Initialize the ScrapegraphToolSpec
scrapegraph_tool = ScrapegraphToolSpec()

# Convert webpage content to markdown
response = scrapegraph_tool.scrapegraph_markdownify(
    url="https://scrapegraphai.com/",
    api_key="sgai-***",  # Replace with your actual API key
)

# Print the markdown content
print("Markdown Content:")
print(response)
