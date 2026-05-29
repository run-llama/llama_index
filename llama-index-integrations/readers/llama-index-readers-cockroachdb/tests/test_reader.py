"""Tests for CockroachDBReader."""

from __future__ import annotations

from typing import Any

import psycopg2
import pytest

from llama_index.readers.cockroachdb import CockroachDBReader


@pytest.fixture()
def seeded_table(fresh_db: dict[str, Any]) -> dict[str, Any]:
    conn = psycopg2.connect(
        host=fresh_db["host"],
        port=fresh_db["port"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        dbname=fresh_db["database"],
        sslmode="disable",
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE articles (
                id INT PRIMARY KEY,
                body STRING NOT NULL,
                author STRING,
                tag STRING
            )
            """
        )
        cur.execute(
            "INSERT INTO articles VALUES (1, 'hello world', 'alice', 'a'), "
            "(2, 'goodbye world', 'bob', 'b'), (3, 'cspann rocks', 'alice', 'a')"
        )
    conn.close()
    return fresh_db


def test_load_data_from_table(seeded_table: dict[str, Any]) -> None:
    reader = CockroachDBReader.from_params(
        host=seeded_table["host"],
        port=seeded_table["port"],
        database=seeded_table["database"],
        user=seeded_table["user"],
        password=seeded_table["password"] or "",
        sslmode="disable",
    )
    docs = reader.load_data(
        table="articles",
        text_column="body",
        metadata_columns=["id", "author", "tag"],
        id_column="id",
    )
    assert len(docs) == 3
    by_id = {d.id_: d for d in docs}
    assert by_id["1"].text == "hello world"
    assert by_id["1"].metadata["author"] == "alice"
    assert set(by_id["2"].metadata) == {"id", "author", "tag"}


def test_load_data_from_query(seeded_table: dict[str, Any]) -> None:
    reader = CockroachDBReader.from_params(
        host=seeded_table["host"],
        port=seeded_table["port"],
        database=seeded_table["database"],
        user=seeded_table["user"],
        password=seeded_table["password"] or "",
        sslmode="disable",
    )
    docs = reader.load_data(
        query="SELECT id, body, author FROM articles WHERE tag = :tag",
        text_column="body",
        metadata_columns=["id", "author"],
        id_column="id",
        params={"tag": "a"},
    )
    assert {d.text for d in docs} == {"hello world", "cspann rocks"}
