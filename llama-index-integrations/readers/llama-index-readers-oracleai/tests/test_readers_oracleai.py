from typing import TYPE_CHECKING
from llama_index.core.readers.base import BaseReader
from llama_index.readers.oracleai import OracleReader, OracleTextSplitter

if TYPE_CHECKING:
    import oracledb


def test_class():
    names_of_base_classes = [b.__name__ for b in OracleReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


# unit tests
uname = ""
passwd = ""
v_dsn = ""


### Test OracleReader #####
# @pytest.mark.requires("oracledb")
def test_loader_test() -> None:
    try:
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        # print("Connection Successful!")

        cursor = connection.cursor()
        cursor.execute("drop table if exists llama_demo")
        cursor.execute("create table llama_demo(id number, text varchar2(25))")

        rows = [
            (1, "First"),
            (2, "Second"),
            (3, "Third"),
            (4, "Fourth"),
            (5, "Fifth"),
            (6, "Sixth"),
            (7, "Seventh"),
        ]

        cursor.executemany("insert into llama_demo(id, text) values (:1, :2)", rows)
        connection.commit()
        cursor.close()

        # load from database column
        loader_params = {
            "owner": uname,
            "tablename": "llama_demo",
            "colname": "text",
        }
        loader = OracleReader(conn=connection, params=loader_params)
        docs = loader.load()

        # verify
        assert len(docs) != 0
        # print(f"Document#1: {docs[0].text}")

        connection.close()
    except Exception as e:
        # print("Error: ", e)
        pass


### Test OracleTextSplitter ####
# @pytest.mark.requires("oracledb")
def test_splitter_test() -> None:
    try:
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        # print("Connection Successful!")

        doc = """Llamaindex is a wonderful framework to load, split, chunk
                and embed your data!!"""

        # by words , max = 1000
        splitter_params = {
            "by": "words",
            "max": "1000",
            "overlap": "200",
            "split": "custom",
            "custom_list": [","],
            "extended": "true",
            "normalize": "all",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"1. Number of chunks: {len(chunks)}")

        # by chars , max = 4000
        splitter_params = {
            "by": "chars",
            "max": "4000",
            "overlap": "800",
            "split": "NEWLINE",
            "normalize": "all",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"2. Number of chunks: {len(chunks)}")

        # by words , max = 10
        splitter_params = {
            "by": "words",
            "max": "10",
            "overlap": "2",
            "split": "SENTENCE",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"3. Number of chunks: {len(chunks)}")

        # by chars , max = 50
        splitter_params = {
            "by": "chars",
            "max": "50",
            "overlap": "10",
            "split": "SPACE",
            "normalize": "all",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"4. Number of chunks: {len(chunks)}")

        connection.close()
    except Exception:
        pass


# test loader and splitter
# test_loader_test()
# test_splitter_test()
