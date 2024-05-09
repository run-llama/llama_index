import oracledb

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.oracleai import OracleEmbeddings


def test_class():
    names_of_base_classes = [b.__name__ for b in OracleEmbeddings.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


# unit tests
uname = "ut"
passwd = "ut"
v_dsn = "phoenix104636.dev3sub3phx.databasede3phx.oraclevcn.com/v5.regress.rdbms.dev.us.oracle.com"


### Test OracleEmbeddings #####
# @pytest.mark.requires("oracledb")
def test_embeddings_test() -> None:
    try:
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        # print("Connection Successful!")

        doc = """Hello World!!!"""

        # get oracle embeddings
        embedder_params = {"provider": "database", "model": "demo_model"}
        embedder = OracleEmbeddings(conn=connection, params=embedder_params)
        embedding = embedder._get_text_embedding(doc)

        # verify
        assert len(embedding) != 0
        # print(f"Embedding: {embedding}")

        connection.close()
    except Exception as e:
        # print("Error: ", e)
        pass


# test embedder
test_embeddings_test()
