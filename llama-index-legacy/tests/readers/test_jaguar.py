import json

from llama_index.legacy.readers.jaguar import JaguarReader
from llama_index.legacy.schema import TextNode
from llama_index.legacy.vector_stores.jaguar import JaguarVectorStore

#############################################################################################
##
##  This test uses JaguarVectorStore and JaguarReader.
##  JaguarVectorStore is responsible for writing test data into the vector store.
##  JaguarReader is responsible for reading (loading) data from the vector store.
##  They are independent objects both of which require login to the vector store
##  and logout from the vector store.
##
##  Requirement: fwww http server must be running at 127.0.0.1:8080 (or any end point)
##               jaguardb server must be running accepting commands from the http server
##
#############################################################################################


class TestJaguarReader:
    vectorstore: JaguarVectorStore
    reader: JaguarReader
    pod: str
    store: str
    mockClient: bool

    @classmethod
    def setup_class(cls) -> None:
        url = "http://127.0.0.1:8080/fwww/"
        cls.pod = "vdb"
        cls.store = "llamaindex_reader_store"
        cls.mockClient = False
        vector_index = "v"
        vector_type = "cosine_fraction_float"
        vector_dimension = 3
        try:
            cls.vectorstore = JaguarVectorStore(
                cls.pod,
                cls.store,
                vector_index,
                vector_type,
                vector_dimension,
                url,
            )

            cls.reader = JaguarReader(
                cls.pod,
                cls.store,
                vector_index,
                vector_type,
                vector_dimension,
                url,
            )
        except ValueError:
            cls.mockClient = True

    @classmethod
    def teardown_class(cls) -> None:
        pass

    def test_login(self) -> None:
        """Client must login to jaguar store server.

        Environment variable JAGUAR_API_KEY or $HOME/.jagrc file must
        contain the jaguar api key
        """
        if self.mockClient:
            return

        rc1 = self.vectorstore.login()
        assert rc1 is True

        rc2 = self.reader.login()
        assert rc2 is True

    def test_create(self) -> None:
        """Create a vector with vector index 'v' of vector_dimension.

        and 'v:text' to hold text and metadata fields author and category
        """
        if self.mockClient:
            return

        metadata_fields = "author char(32), category char(16)"
        self.vectorstore.create(metadata_fields, 1024)

        ### verify the table is created correctly
        podstore = self.pod + "." + self.store
        js = self.vectorstore.run(f"desc {podstore}")
        jd = json.loads(js[0])
        assert podstore in jd["data"]

    def test_add_texts(self) -> None:
        """Add some text nodes through vectorstore."""
        if self.mockClient:
            return

        self.vectorstore.clear()

        node1 = TextNode(
            text="Return of King Lear",
            metadata={"author": "William", "category": "Tragedy"},
            embedding=[0.9, 0.1, 0.4],
        )

        node2 = TextNode(
            text="Slow Clouds",
            metadata={"author": "Adam", "category": "Nature"},
            embedding=[0.4, 0.2, 0.8],
        )

        node3 = TextNode(
            text="Green Machine",
            metadata={"author": "Eve", "category": "History"},
            embedding=[0.1, 0.7, 0.5],
        )

        nodes = [node1, node2, node3]

        ids = self.vectorstore.add(nodes=nodes, use_node_metadata=True)
        assert len(ids) == len(nodes)
        assert len(ids) == 3

    def test_query_embedding(self) -> None:
        """Test that [0.4, 0.2, 0.8] will retrieve Slow Clouds.

        This test case uses similarity search.
        Here k is 1.
        """
        if self.mockClient:
            return

        embed = [0.4, 0.2, 0.8]
        fields = ["author", "category"]
        docs = self.reader.load_data(embedding=embed, k=1, metadata_fields=fields)

        assert len(docs) == 1
        assert docs[0].text == "Slow Clouds"
        assert docs[0].metadata["author"] == "Adam"
        assert docs[0].metadata["category"] == "Nature"

    def test_query_data_limit(self) -> None:
        """Test query date of 2 records."""
        if self.mockClient:
            return

        fields = ["author", "category"]
        docs = self.reader.load_data(k=2, metadata_fields=fields)
        assert len(docs) == 2

    def test_query_data_filter(self) -> None:
        """Test query date with filter(where condition)."""
        if self.mockClient:
            return

        fields = ["author", "category"]
        where = "author='Eve' or author='Charles'"
        docs = self.reader.load_data(k=1, metadata_fields=fields, where=where)

        assert len(docs) == 1
        assert docs[0].text == "Green Machine"
        assert docs[0].metadata["author"] == "Eve"
        assert docs[0].metadata["category"] == "History"

    def test_clear(self) -> None:
        """Test cleanup of data in the store."""
        if self.mockClient:
            return

        self.vectorstore.clear()
        assert self.vectorstore.count() == 0

    def test_drop(self) -> None:
        """Destroy the vector store."""
        if self.mockClient:
            return

        self.vectorstore.drop()

    def test_logout(self) -> None:
        """Client must logout to disconnect from jaguar server.

        and clean up resources used by the client
        """
        if self.mockClient:
            return

        self.vectorstore.logout()
        self.reader.logout()
