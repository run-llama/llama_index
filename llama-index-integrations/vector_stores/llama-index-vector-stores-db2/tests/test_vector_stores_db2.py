from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.db2 import (
    DB2LlamaVS,
    create_table,
    table_exists,
    drop_table,
    DistanceStrategy,
)

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in DB2LlamaVS.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


database = ""
username = ""
password = ""


def test_table_exists() -> None:
    try:
        import ibm_db_dbi  # type: ignore
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        return

    # 1. Create a Table
    create_table(connection, "TB1", 8148)

    # 2. Existing Table
    # expectation:true
    assert table_exists(connection, "TB1")

    # 3. Non-Existing Table
    # expectation:false
    assert not table_exists(connection, "TableNonExist")

    # 4. Invalid Table Name
    # Expectation: SQL0104N
    try:
        table_exists(connection, "123")
    except Exception:
        pass

    # 5. Empty String
    # Expectation: SQL0104N
    try:
        table_exists(connection, "")
    except Exception:
        pass

    # 6. Special Character
    # Expectation: SQL0007N
    try:
        table_exists(connection, "!!4")
    except Exception:
        pass

    # 7. Table name length > 128
    # Expectation: SQL0107N The name is too long.  The maximum length is "128".
    try:
        table_exists(connection, "x" * 129)
    except Exception:
        pass

    # 8. Toggle Upper/Lower Case (like TaBlE)
    # Expectation:True
    assert table_exists(connection, "Tb1")
    drop_table(connection, "TB1")

    # 9. Table_Name→ "表格"
    # Expectation:True
    create_table(connection, '"表格"', 545)
    assert table_exists(connection, '"表格"')
    drop_table(connection, '"表格"')

    connection.commit()


def test_create_table() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        return

    # 1. New table - HELLO
    #    Dimension - 100
    # Expectation: table is created
    create_table(connection, "HELLO", 100)

    # 2. Existing table name - HELLO
    #    Dimension - 110
    # Expectation: Log message table already exists
    create_table(connection, "HELLO", 110)
    drop_table(connection, "HELLO")

    # 3. New Table - 123
    #    Dimension - 100
    # Expectation: SQL0104N  invalid table name
    try:
        create_table(connection, "123", 100)
        drop_table(connection, "123")
    except Exception:
        pass

    # 4. New Table - Hello123
    #    Dimension - 8148
    # Expectation: table is created
    create_table(connection, "Hello123", 8148)
    drop_table(connection, "Hello123")

    # 5. New Table - T1
    #    Dimension - 65536
    # Expectation: SQL0604N  VECTOR column exceed the supported
    # dimension length.
    try:
        create_table(connection, "T1", 65536)
        drop_table(connection, "T1")
    except Exception:
        pass

    # 6. New Table - T1
    #    Dimension - 0
    # Expectation: SQL0604N  VECTOR column unsupported dimension length 0.
    try:
        create_table(connection, "T1", 0)
        drop_table(connection, "T1")
    except Exception:
        pass

    # 7. New Table - T1
    #    Dimension - -1
    # Expectation: SQL0104N  An unexpected token "-" was found
    try:
        create_table(connection, "T1", -1)
        drop_table(connection, "T1")
    except Exception:
        pass

    # 8. New Table - T2
    #     Dimension - '1000'
    # Expectation: table is created
    create_table(connection, "T2", int("1000"))
    drop_table(connection, "T2")

    # 9. New Table - T3
    #     Dimension - 100 passed as a variable
    # Expectation: table is created
    val = 100
    create_table(connection, "T3", val)
    drop_table(connection, "T3")

    # 10.
    # Expectation: SQL0104N  An unexpected token
    val2 = """H
    ello"""
    try:
        create_table(connection, val2, 545)
        drop_table(connection, val2)
    except Exception:
        pass

    # 11. New Table - 表格
    #     Dimension - 545
    # Expectation: table is created
    create_table(connection, '"表格"', 545)
    drop_table(connection, '"表格"')

    # 12. <schema_name.table_name>
    # Expectation: table with schema is created
    create_table(connection, "U1.TB4", 128)
    drop_table(connection, "U1.TB4")

    # 13.
    # Expectation: table is created
    create_table(connection, '"T5"', 128)
    drop_table(connection, '"T5"')

    # 14. Toggle Case
    # Expectation: table is created
    create_table(connection, "TaBlE", 128)
    drop_table(connection, "TaBlE")

    # 15. table_name as empty_string
    # Expectation: SQL0104N  An unexpected token
    try:
        create_table(connection, "", 128)
        drop_table(connection, "")
        create_table(connection, '""', 128)
        drop_table(connection, '""')
    except Exception:
        pass

    # 16. Arithmetic Operations in dimension parameter
    # Expectation: table is created
    n = 1
    create_table(connection, "T10", n + 500)
    drop_table(connection, "T10")

    # 17. String Operations in table_name parameter
    # Expectation: table is created
    create_table(connection, "YaSh".replace("aS", "ok"), 500)
    drop_table(connection, "YaSh".replace("aS", "ok"))

    connection.commit()


# Define a list of documents (These dummy examples are 4 random documents )
text_json_list = [
    {
        "text": "Db2 handles LOB data differently than other kinds of data. As a result, you sometimes need to take additional actions when you define LOB columns and insert the LOB data.",
        "id_": "doc_1_2_P4",
        "embedding": [1.0, 0.0],
        "relationships": "test-0",
        "metadata": {
            "weight": 1.0,
            "rank": "a",
            "url": "https://www.ibm.com/docs/en/db2-for-zos/12?topic=programs-storing-lob-data-in-tables",
        },
    },
    {
        "text": "Introduced in Db2 13, SQL Data Insights brought artificial intelligence (AI) functionality to the Db2 for z/OS engine. It provided the capability to run SQL AI query to find valuable insights hidden in your Db2 data and help you make better business decisions.",
        "id_": "doc_15.5.1_P1",
        "embedding": [0.0, 1.0],
        "relationships": "test-1",
        "metadata": {
            "weight": 2.0,
            "rank": "c",
            "url": "https://community.ibm.com/community/user/datamanagement/blogs/neena-cherian/2023/03/07/accelerating-db2-ai-queries-with-the-new-vector-pr",
        },
    },
    {
        "text": "Data structures are elements that are required to use DB2®. You can access and use these elements to organize your data. Examples of data structures include tables, table spaces, indexes, index spaces, keys, views, and databases.",
        "id_": "id_22.3.4.3.1_P2",
        "embedding": [1.0, 1.0],
        "relationships": "test-2",
        "metadata": {
            "weight": 3.0,
            "rank": "d",
            "url": "https://www.ibm.com/docs/en/zos-basic-skills?topic=concepts-db2-data-structures",
        },
    },
    {
        "text": "DB2® maintains a set of tables that contain information about the data that DB2 controls. These tables are collectively known as the catalog. The catalog tables contain information about DB2 objects such as tables, views, and indexes. When you create, alter, or drop an object, DB2 inserts, updates, or deletes rows of the catalog that describe the object.",
        "id_": "id_3.4.3.1_P3",
        "embedding": [2.0, 1.0],
        "relationships": "test-3",
        "metadata": {
            "weight": 4.0,
            "rank": "e",
            "url": "https://www.ibm.com/docs/en/zos-basic-skills?topic=objects-db2-catalog",
        },
    },
]

# Create Llama Text Nodes
text_nodes = []
for text_json in text_json_list:
    # Construct the relationships using RelatedNodeInfo
    relationships = {
        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=text_json["relationships"])
    }

    # Prepare the metadata dictionary; you might want to exclude certain metadata fields if necessary
    metadata = {
        "weight": text_json["metadata"]["weight"],
        "rank": text_json["metadata"]["rank"],
    }

    # Create a TextNode instance
    text_node = TextNode(
        text=text_json["text"],
        id_=text_json["id_"],
        embedding=text_json["embedding"],
        relationships=relationships,
        metadata=metadata,
    )

    text_nodes.append(text_node)


vector_store_list = []


def test_vs_creation() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        return

    # Ingest documents into Db2 Vector Store using different distance strategies
    vector_store_dot = DB2LlamaVS.from_documents(
        text_nodes,
        table_name="Documents_DOT",
        client=connection,
        distance_strategy=DistanceStrategy.DOT_PRODUCT,
        embed_dim=2,
    )
    vector_store_list.append(vector_store_dot)
    vector_store_max = DB2LlamaVS.from_documents(
        text_nodes,
        table_name="Documents_COSINE",
        client=connection,
        distance_strategy=DistanceStrategy.COSINE,
        embed_dim=2,
    )
    vector_store_list.append(vector_store_max)
    vector_store_euclidean = DB2LlamaVS.from_documents(
        text_nodes,
        table_name="Documents_EUCLIDEAN",
        client=connection,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        embed_dim=2,
    )
    vector_store_list.append(vector_store_euclidean)

    connection.commit()


def test_manage_texts():
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        return

    for i, vs in enumerate(vector_store_list, start=1):
        # Adding texts
        try:
            vs.add_texts(text_nodes, metadata)
            print(f"\n\n\nAdd texts complete for vector store {i}\n\n\n")
        except Exception as ex:
            print(f"\n\n\nExpected error on duplicate add for vector store {i}\n\n\n")

        # Deleting texts using the value of 'doc_id'
        vs.delete("test-1")
        print(f"\n\n\nDelete texts complete for vector store {i}\n\n\n")

        # Similarity search
        query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
        results = vs.query(query=query)
        print(f"\n\n\nSimilarity search results for vector store {i}: {results}\n\n\n")

    connection.commit()


def test_advanced_searches():
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        return

    for i, vs in enumerate(vector_store_list, start=1):

        def query_without_filters_returns_all_rows_sorted_by_similarity():
            print(f"\n--- Vector Store {i} Advanced Searches ---")
            # Similarity search without a filter
            print("\nSimilarity search results without filter:")
            query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
            print(vs.query(query=query))

        query_without_filters_returns_all_rows_sorted_by_similarity()

        def query_with_filters_returns_multiple_matches():
            print(f"\n--- Vector Store {i} Advanced Searches ---")
            # Similarity search with filter
            print("\nSimilarity search results with filter:")
            filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
            query = VectorStoreQuery(
                query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=3
            )
            result = vs.query(query)
            print(result.ids)

        query_with_filters_returns_multiple_matches()

        def query_with_filter_applies_top_k():
            print(f"\n--- Vector Store {i} Advanced Searches ---")
            # Similarity search with a filter
            print("\nSimilarity search results with top k filter:")
            filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
            query = VectorStoreQuery(
                query_embedding=[1.0, 1.0], filters=filters, similarity_top_k=1
            )
            result = vs.query(query)
            print(result.ids)

        query_with_filter_applies_top_k()

        def query_with_filter_applies_node_id_filter():
            print(f"\n--- Vector Store {i} Advanced Searches ---")
            # Similarity search with a filter
            print("\nSimilarity search results with node_id filter:")
            filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
            query = VectorStoreQuery(
                query_embedding=[1.0, 1.0],
                filters=filters,
                similarity_top_k=3,
                node_ids=["452D24AB-F185-414C-A352-590B4B9EE51B"],
            )
            result = vs.query(query)
            print(result.ids)

        query_with_filter_applies_node_id_filter()

        def query_with_exact_filters_returns_single_match():
            print(f"\n--- Vector Store {i} Advanced Searches ---")
            # Similarity search with a filter
            print("\nSimilarity search results with filter:")
            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="rank", value="c"),
                    ExactMatchFilter(key="weight", value=2),
                ]
            )
            query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
            result = vs.query(query)
            print(result.ids)

        query_with_exact_filters_returns_single_match()

        def query_with_contradictive_filter_returns_no_matches():
            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="weight", value=2),
                    ExactMatchFilter(key="weight", value=3),
                ]
            )
            query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
            result = vs.query(query)
            print(result.ids)

        query_with_contradictive_filter_returns_no_matches()

        def query_with_filter_on_unknown_field_returns_no_matches():
            print(f"\n--- Vector Store {i} Advanced Searches ---")
            # Similarity search with a filter
            print("\nSimilarity search results with filter:")
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="unknown_field", value="c")]
            )
            query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
            result = vs.query(query)
            print(result.ids)

        query_with_filter_on_unknown_field_returns_no_matches()

        def delete_removes_document_from_query_results():
            vs.delete("test-1")
            query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=2)
            result = vs.query(query)
            print(result.ids)

        delete_removes_document_from_query_results()

    connection.commit()
