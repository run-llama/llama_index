# LlamaIndex Readers Integration: Couchbase

```bash
pip install llama-index-readers-couchbase
```

This loader loads documents from Couchbase cluster.
The user specifies a Couchbase client or credentials to initialize the reader. They can specify the SQL++ query to
fetch the relevant docs.

## Usage

Here's an example usage of the CouchbaseReader.

```python
import os

from llama_index.readers.couchbase import CouchbaseReader

connection_string = (
    "couchbase://localhost"  # valid Couchbase connection string
)
db_username = "<valid_database_user_with_read_access_to_bucket_with_data>"
db_password = "<password_for_database_user>"

# query is a valid SQL++ query that is passed to client.query()
query = """
    SELECT h.* FROM `travel-sample`.inventory.hotel h
        WHERE h.country = 'United States'
        LIMIT 5
        """

reader = CouchbaseLoader(
    connection_string=connection_string,
    db_username=db_username,
    db_password=db_password,
)

# It is also possible to pass an initialized Couchbase client to the document loader
# from couchbase.auth import PasswordAuthenticator  # noqa: E402
# from couchbase.cluster import Cluster # noqa: E402
# from couchbase.options import ClusterOptions # noqa: E402

# auth = PasswordAuthenticator(
#    db_username,
#    db_password,
# )

# couchbase_client = Cluster(connection_string, ClusterOptions(auth))
# reader = CouchbaseLoader(client=couchbase_client)

# fields to be written to the document
text_fields = ["name", "title", "address", "reviews"]

# metadata fields to be written to the document's metadata
metadata_fields = (["country", "city"],)

documents = reader.load_data(
    query=query, text_fields=text_fields, metadata_fields=metadata_fields
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/). See [here](https://github.com/run-llama/llama-hub/tree/main) for examples.
