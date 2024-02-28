# BoardDocs Loader

This loader retrieves an agenda and associated material from a BoardDocs site.

This loader is not endorsed by, developed by, supported by, or in any way formally affiliated with Diligent Corporation.

## Usage

To use this loader, you'll need to specify which BoardDocs site you want to load,
as well as the committee on the site you want to scrape.

```python
from llama_index import download_loader

BoardDocsReader = download_loader("BoardDocsReader")

# For a site URL https://go.boarddocs.com/ca/redwood/Board.nsf/Public
# your site should be set to 'ca/redwood'
# You'll also need to specify which committee on the site you want to index,
# in this case A4EP6J588C05 is the Board of Trustees meeting.
loader = BoardDocsReader(site="ca/redwood", committee_id="A4EP6J588C05")

# You can optionally specify to load a specific set of meetings; if you don't
# pass in meeting_ids, the loader will attempt to load *all* meeting content.
# Since we're actually scraping a site, this can take a little while.
documents = loader.load_data(meeting_ids=["CPSNV9612DF1"])
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
