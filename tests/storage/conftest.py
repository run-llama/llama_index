from pathlib import Path
import pytest
from gpt_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore
from gpt_index.storage.kvstore.simple_kvstore import SimpleKVStore
from tests.storage.kvstore.mock_mongodb import MockMongoClient


@pytest.fixture()
def mongo_client() -> MockMongoClient:
    return MockMongoClient()


@pytest.fixture()
def mongo_kvstore(mongo_client: MockMongoClient) -> MongoDBKVStore:
    return MongoDBKVStore(mongo_client=mongo_client)  # type: ignore


@pytest.fixture()
def simple_kvstore(tmp_path: Path) -> SimpleKVStore:
    file_path = str(tmp_path / "test_file.txt")
    return SimpleKVStore(file_path)
