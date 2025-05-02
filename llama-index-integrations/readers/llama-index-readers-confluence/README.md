# Confluence Loader

```bash
pip install llama-index-readers-confluence
```

This loader loads pages from a given Confluence cloud instance. The user needs to specify the base URL for a Confluence
instance to initialize the ConfluenceReader - base URL needs to end with `/wiki`.

The user can optionally specify OAuth 2.0 credentials to authenticate with the Confluence instance. If no credentials are
specified, the loader will look for `CONFLUENCE_API_TOKEN` or `CONFLUENCE_USERNAME`/`CONFLUENCE_PASSWORD` environment variables
to proceed with basic authentication.

> [!NOTE]
> Keep in mind `CONFLUENCE_PASSWORD` is not your actual password, but an API Token obtained here: https://id.atlassian.com/manage-profile/security/api-tokens.

The following order is used for checking authentication credentials:

1. `oauth2`
2. `api_token`
3. `cookies`
4. `user_name` and `password`
5. Environment variable `CONFLUENCE_API_TOKEN`
6. Environment variable `CONFLUENCE_USERNAME` and `CONFLUENCE_PASSWORD`

For more on authenticating using OAuth 2.0, checkout:

- https://atlassian-python-api.readthedocs.io/index.html
- https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/

Confluence pages are obtained through one of 4 four mutually exclusive ways:

1. `page_ids`: Load all pages from a list of page ids
2. `space_key`: Load all pages from a space
3. `label`: Load all pages with a given label
4. `cql`: Load all pages that match a given CQL query (Confluence Query Language https://developer.atlassian.com/cloud/confluence/advanced-searching-using-cql/ ).

When `page_ids` is specified, `include_children` will cause the loader to also load all descendent pages.
When `space_key` is specified, `page_status` further specifies the status of pages to load: None, 'current', 'archived', 'draft'.

limit (int): Deprecated, use `max_num_results` instead.

max_num_results (int): Maximum number of results to return. If None, return all results. Requests are made in batches to achieve the desired number of results.

start(int): Which offset we should jump to when getting pages, only works with space_key

cursor(str): An alternative to start for cql queries, the cursor is a pointer to the next "page" when searching atlassian products. The current one after a search can be found with `get_next_cursor()`

User can also specify a boolean `include_attachments` to
include attachments, this is set to `False` by default, if set to `True` all attachments will be downloaded and
ConfluenceReader will extract the text from the attachments and add it to the Document object.
Currently supported attachment types are: PDF, PNG, JPEG/JPG, SVG, Word and Excel.

Hint: `space_key` and `page_id` can both be found in the URL of a page in Confluence - https://yoursite.atlassian.com/wiki/spaces/<space_key>/pages/<page_id>

## Usage

Here's an example usage of the ConfluenceReader.

```python
# Example that reads the pages with the `page_ids`
from llama_index.readers.confluence import ConfluenceReader

token = {"access_token": "<access_token>", "token_type": "<token_type>"}
oauth2_dict = {"client_id": "<client_id>", "token": token}

base_url = "https://yoursite.atlassian.com/wiki"

page_ids = ["<page_id_1>", "<page_id_2>", "<page_id_3"]
space_key = "<space_key>"

reader = ConfluenceReader(
    base_url=base_url,
    oauth2=oauth2_dict,
    client_args={"backoff_and_retry": True},
)
documents = reader.load_data(
    space_key=space_key, include_attachments=True, page_status="current"
)
documents.extend(
    reader.load_data(
        page_ids=page_ids, include_children=True, include_attachments=True
    )
)
```

```python
# Example that fetches the first 5, then the next 5 pages from a space
from llama_index.readers.confluence import ConfluenceReader

token = {"access_token": "<access_token>", "token_type": "<token_type>"}
oauth2_dict = {"client_id": "<client_id>", "token": token}

base_url = "https://yoursite.atlassian.com/wiki"

space_key = "<space_key>"

reader = ConfluenceReader(base_url=base_url, oauth2=oauth2_dict)
documents = reader.load_data(
    space_key=space_key,
    include_attachments=True,
    page_status="current",
    start=0,
    max_num_results=5,
)
documents.extend(
    reader.load_data(
        space_key=space_key,
        include_children=True,
        include_attachments=True,
        start=5,
        max_num_results=5,
    )
)
```

```python
# Example that fetches the first 5 results from a cql query, the uses the cursor to pick up on the next element
from llama_index.readers.confluence import ConfluenceReader

token = {"access_token": "<access_token>", "token_type": "<token_type>"}
oauth2_dict = {"client_id": "<client_id>", "token": token}

base_url = "https://yoursite.atlassian.com/wiki"

cql = f'type="page" AND label="devops"'

reader = ConfluenceReader(base_url=base_url, oauth2=oauth2_dict)
documents = reader.load_data(cql=cql, max_num_results=5)
cursor = reader.get_next_cursor()
documents.extend(reader.load_data(cql=cql, cursor=cursor, max_num_results=5))
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
