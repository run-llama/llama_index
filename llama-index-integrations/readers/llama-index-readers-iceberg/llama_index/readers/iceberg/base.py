from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import Any, Dict, List, Optional, Tuple

from pyiceberg.catalog import load_catalog
import json


class IcebergReader(BaseReader):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_columns(
            self, all_columns:List[str], metadata_columns_in: List[str]
    ) -> Tuple[List[str], List[str]]:
        content_columns = []
        metadata_columns = []
        for key in all_columns:
            if key in metadata_columns_in:
                metadata_columns.append(key)
            else:
                content_columns.append(key)

        return content_columns, metadata_columns

    def load_data(
            self,
            namespace: str,
            table: str,
            profile_name: str = "default",
            region: str = "us-east-1",
            metadata_columns: Optional[List[str]] = None,
            extra_info: Optional[Dict] = None,
    ) -> List[Document]:

        metadata_columns = metadata_columns if metadata_columns is not None else []
        underlying_db = load_catalog("glue",
                                     **{"type": "glue",
                                        "s3.region": region,
                                        "profile_name": profile_name
                                        })
        underlying_table = underlying_db.load_table(f"{namespace}.{table}")
        df = underlying_table.scan().to_pandas()
        query_result = json.loads(df.to_json(orient="records"))
        content_columns, meta_columns = self._get_columns(all_columns=df.columns.tolist(), metadata_columns_in=metadata_columns)
        documents = []
        for row in query_result:
            text = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in content_columns
            )
            metadata = {
                k: v for k, v in row.items() if k in meta_columns and v is not None
            }
            documents.append(Document(text=text, metadata=metadata))

        return documents