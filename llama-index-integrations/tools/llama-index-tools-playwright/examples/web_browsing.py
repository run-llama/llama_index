from llama_index.tools.playwright.base import PlaywrightToolSpec
import time

# Create Playwright browser and tool object
browser = PlaywrightToolSpec.create_sync_playwright_browser(headless=False)
playwright_tool = PlaywrightToolSpec.from_sync_browser(browser)

# List all tools
playwright_tool_list = playwright_tool.to_tool_list()
for tool in playwright_tool_list:
    print(tool.metadata.name)

# Navigate to the playwright doc website
playwright_tool.navigate_to("https://playwright.dev/python/docs/intro")
time.sleep(1)

# Print the current page URL
print(playwright_tool.get_current_page())

# Extract all hyperlinks
print(playwright_tool.extract_hyperlinks())

# Extract all text
print(playwright_tool.extract_text())

# Get element attributes for navigating to the next page
# You can retrieve the selector from google chrome dev tools
element = playwright_tool.get_elements(
    selector='#__docusaurus_skipToContent_fallback > div > div > main > div > div > div.col.docItemCol_VOVn > div > nav > a',
    attributes=['innerText']
)
print(element)

# Click on the search bar
playwright_tool.click(selector='#__docusaurus > nav > div.navbar__inner > div.navbar__items.navbar__items--right > div.navbarSearchContainer_Bca1 > button')
time.sleep(1)

# Fill in the search bar
playwright_tool.fill(selector='#docsearch-input', value='Hello')
time.sleep(1)
