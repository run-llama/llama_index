# LlamaIndex Vector_Stores Integration: Lindrom
Please refer to the [notebook](../../../docs/docs/examples/vector_stores/LindormSearchDemo.ipynb) for usage of Lindorm as vector store in LlamaIndex.

# Example Usage

```sh
pip install llama-index
pip install opensearch-py
pip install llama-index-vector-stores-lindorm
```

```python
from llama_index.vector_stores.lindorm import (
      LindormSearchVectorStore,
      LindormSearchVectorClient,
)

# lindorm instance info
# how to obtain an lindorm instance:
# https://alibabacloud.com/help/en/lindorm/latest/create-an-instance?spm=a2c63.l28256.0.0.4cc0f53cUfKOxI 

# how to access your lindorm instance:
# https://www.alibabacloud.com/help/en/lindorm/latest/view-endpoints?spm=a2c63.p38356.0.0.37121bcdxsDvbN

# run curl commands to connect to and use LindormSearch:
# https://www.alibabacloud.com/help/en/lindorm/latest/connect-and-use-the-search-engine-with-the-curl-command

# host and port of your instance
host = 'ld-bp******jm*******-proxy-search-pub.lindorm.aliyuncs.com'
port = 30070

# username and password of your instance
username = "your_username"
passsword = "your_password"

# index to demonstrate the VectorStore impl
index_name = "lindorm_test_index"

# LindormSearchVectorClient encapsulates logic for a single index with vector search enabled
client = LindormSearchClient(
    host=host,
    port=30070,
    username=username,
    password=password,
    index=index_name,
    dimension=5,
)

# initialize vector store
vector_store = LindormSearchVectorStore(client)


"""-------test for lindormsearch vector store--------"""

print("\n-----------1. add nodes to index test---------\n")
nodes = []
for i in range(0, 1000):
    # Generate a random vector
    vector = [random.random() for _ in range(0, 5)]
    # Generate a random text
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=5))
    # append node to nodes list
    new_node = TextNode(
        embedding = vector,
        text = random_string,
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-"+str(i))},
        metadata={
            "author": "test"+ " " + random_string,
            "mark_id": i,
        }
    )
    nodes.append(new_node)
print(vector_store.add(nodes))

print("\n---------------2. query test-----------------\n")
vector = [1.0, 1.0, 1.0, 1.0, 1.0]

# 2.1 simple query
print("\n-----------------simple query----------------\n")
simple_query = VectorStoreQuery(query_embedding=vector,similarity_top_k=5)
print(vector_store.query(simple_query))


# 2.2 query with post-filter
print("\n-----------query with metadatafilter---------\n")
# range filter test
filter1 = MetadataFilter(key="metadata.mark_id",value=0,operator=FilterOperator.GTE)
filter2 = MetadataFilter(key="metadata.mark_id",value=500,operator=FilterOperator.LTE)
filters = MetadataFilters(filters=[filter1,filter2],condition=FilterCondition.AND)

# term filter test
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="metadata.author",
            value="one author name",
            operator=FilterOperator.EQ
            )
        ],
        condition=FilterCondition.AND
    )

filter_query = VectorStoreQuery(
    query_embedding=vector,
    similarity_top_k=5, 
    filters=filters
)
print(vector_store.query(filter_query))

# 2.3 lexical query
print("\n-----------lexical_search query---------\n")
legal_query = VectorStoreQuery(
    mode=VectorStoreQueryMode.TEXT_SEARCH,
    query_embedding=vector,
    similarity_top_k=5, 
    # your query str match the field "content"(text you stored in),
    # and note the the minimum search granularity of query str is one token.
    query_str="your query str"
)
print(vector_store.query(legal_query))

# 2.4 hybrid query
print("\n-----------hybrid_search query---------\n")
hybrid_query = VectorStoreQuery(
    mode=VectorStoreQueryMode.HYBRID,
    query_embedding=vector,
    similarity_top_k=5, 
    # your query str match the field "content"(text you stored in),
    # and note the the minimum search granularity of query str is one token.
    query_str="your query str"
)
print(vector_store.query(hybrid_query))


print("\n----3. delete nodes using a ref_doc_id-----\n")
vector_store.delete(ref_doc_id="test-0")
```