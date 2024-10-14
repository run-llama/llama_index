import os

from llama_index.vector_stores.oceanbase import OceanBaseVectorStore


def test_class():
    from pyobvector import ObVecClient

    client = ObVecClient(
        uri=os.getenv("OB_URI", "127.0.0.1:2881"),
        user=os.getenv("OB_USER", "root@test"),
        password=os.getenv("OB_PWD", ""),
        db_name=os.getenv("OB_DBNAME", "test"),
    )

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1024,
    )
    ob = OceanBaseVectorStore()
