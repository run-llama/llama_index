import os
import pytest
from typing import Iterable

import astrapy
from astrapy.db import AstraDB
from llama_index.readers.astra_db import AstraDBReader
from llama_index.core.schema import Document

COLLECTION_NAME = "li_readers_test"

print(f"astrapy detected: {astrapy.__version__}")


# env variables
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")


@pytest.fixture(autouse=True, scope="module")
def source_collection() -> Iterable[astrapy.db.AstraDBCollection]:
    database = AstraDB(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
    )
    collection = database.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=2,
    )

    collection.insert_many(
        documents=[
            {
                "_id": "doc0",
                "content": "Content 0",
                "$vector": [0.0, 10.0],
            },
            {
                "_id": "doc1",
                "content": "Content 1",
                "$vector": [1.0, 10.0],
            },
            {
                "_id": "doc2",
                "content": "Content 2",
                "$vector": [2.0, 10.0],
            },
        ]
    )

    yield collection

    database.delete_collection(COLLECTION_NAME)


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_astra_db_readers_read() -> None:
    reader = AstraDBReader(
        collection_name=COLLECTION_NAME,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        embedding_dimension=2,
    )
    loaded = reader.load_data(vector=[1, 1])
    assert len(loaded) == 3
    assert loaded[0] == Document(
        doc_id="doc2",
        text="Content 2",
        embedding=[2.0, 10.0],
    )
