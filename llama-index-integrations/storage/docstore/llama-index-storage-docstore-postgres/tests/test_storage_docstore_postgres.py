from llama_index.storage.docstore.postgres import PostgresDocumentStore


def test_from_params_uses_compatible_kvstore() -> None:
    store = PostgresDocumentStore.from_params(
        database="database",
        host="host",
        password="password",
        port="5432",
        user="user",
        perform_setup=False,
    )

    assert isinstance(store, PostgresDocumentStore)
