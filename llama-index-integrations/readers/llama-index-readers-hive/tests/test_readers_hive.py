import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.hive.base import InvalidSqlError, _validate_sql_query
from llama_index.readers.hive import HiveReader


def test_class():
    names_of_base_classes = [b.__name__ for b in HiveReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_validation():
    with pytest.raises(InvalidSqlError):
        sql_query = _validate_sql_query(["SELECT * FROM users", "DROP TABLE users"])
    with pytest.raises(InvalidSqlError):
        sql_query = _validate_sql_query(
            ["SELECT * FROM users WHERE name = 'Bob' OR '1'='1'"]
        )
    with pytest.raises(InvalidSqlError):
        sql_query = _validate_sql_query(
            [
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT,
                    name STRING,
                    email STRING
                )
                STORED AS TEXTFILE
                """
            ]
        )
    sql_query = _validate_sql_query(["SELECT * FROM users WHERE name = 'Bob'"])
    assert sql_query is None
