import json
import os
from pathlib import PurePath
from typing import Any, Dict, Optional

from llama_index.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore

IMPORT_ERROR_MSG = "`boto3` package not found, please run `pip install boto3`"


class S3DBKVStore(BaseKVStore):
    """S3 Key-Value store.
    Stores key-value pairs in a S3 bucket. Can optionally specify a path to a folder
        where KV data is stored.
    The KV data is further divided into collections, which are subfolders in the path.
    Each key-value pair is stored as a JSON file.

    Args:
        s3_bucket (Any): boto3 S3 Bucket instance
        path (Optional[str]): path to folder in S3 bucket where KV data is stored
    """

    def __init__(
        self,
        bucket: Any,
        path: Optional[str] = "./",
    ) -> None:
        """Init a S3DBKVStore."""
        try:
            pass
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._bucket = bucket
        self._path = path or "./"

    @classmethod
    def from_s3_location(
        cls,
        bucket_name: str,
        path: Optional[str] = None,
    ) -> "S3DBKVStore":
        """Load a S3DBKVStore from a S3 URI.

        Args:
            bucket_name (str): S3 bucket name
            path (Optional[str]): path to folder in S3 bucket where KV data is stored
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_name)
        return cls(
            bucket,
            path=path,
        )

    def _get_object_key(self, collection: str, key: str) -> str:
        return str(PurePath(f"{self._path}/{collection}/{key}.json"))

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        obj_key = self._get_object_key(collection, key)
        self._bucket.put_object(
            Key=obj_key,
            Body=json.dumps(val),
        )

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        obj_key = self._get_object_key(collection, key)
        try:
            obj = next(iter(self._bucket.objects.filter(Prefix=obj_key).limit(1)))
        except StopIteration:
            return None
        body = obj.get()["Body"].read()
        return json.loads(body)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): collection name

        """
        collection_path = str(PurePath(f"{self._path}/{collection}/"))
        collection_kv_dict = {}
        for obj in self._bucket.objects.filter(Prefix=collection_path):
            body = obj.get()["Body"].read()
            json_filename = os.path.split(obj.key)[-1]
            key = os.path.splitext(json_filename)[0]
            value = json.loads(body)
            collection_kv_dict[key] = value
        return collection_kv_dict

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        obj_key = self._get_object_key(collection, key)
        matched_objs = list(self._bucket.objects.filter(Prefix=obj_key).limit(1))
        if len(matched_objs) == 0:
            return False
        obj = matched_objs[0]
        obj.delete()
        return True
