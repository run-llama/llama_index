from typing import TYPE_CHECKING
from llama_index.utils.oracleai import OracleSummary

if TYPE_CHECKING:
    import oracledb


# unit tests
uname = ""
passwd = ""
v_dsn = ""


### Test OracleSummary #####
# @pytest.mark.requires("oracledb")
def test_summary_test() -> None:
    try:
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        # print("Connection Successful!")

        doc = """LlamaIndex is a data framework designed specifically
        for Large Language Models (LLMs). It acts as a bridge between
        your enterprise data and LLM applications, allowing you to leverage
        the power of LLMs for various tasks. Here's a breakdown of its key
        features and functionalities: Data Integration, Knowledge Base Creation,
        Retrieval and Augmentation, Integration with LLMs and so on. """

        # get oracle summary
        summary_params = {
            "provider": "database",
            "glevel": "S",
            "numParagraphs": 1,
            "language": "english",
        }
        summary = OracleSummary(conn=connection, params=summary_params)
        summ = summary.get_summary(doc)

        # verify
        assert len(summ) != 0
        # print(f"Summary: {summ}")

        connection.close()
    except Exception as e:
        # print("Error: ", e)
        pass


# test embedder
# test_summary_test()
