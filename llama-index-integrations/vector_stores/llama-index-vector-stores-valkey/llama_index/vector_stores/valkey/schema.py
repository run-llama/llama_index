from dataclasses import dataclass
from typing import Any, List

from glide_shared.commands.server_modules.ft_options.ft_create_options import (
    DataType,
    DistanceMetricType,
    Field,
    TagField,
    TextField,
    VectorAlgorithm,
    VectorField,
    VectorFieldAttributesFlat,
    VectorType,
)

# required llama index fields
NODE_ID_FIELD_NAME: str = "id"
DOC_ID_FIELD_NAME: str = "doc_id"
TEXT_FIELD_NAME: str = "text"
NODE_CONTENT_FIELD_NAME: str = "_node_content"
VECTOR_FIELD_NAME: str = "vector"

# Default embedding dimension (OpenAI text-embedding-ada-002)
DEFAULT_EMBEDDING_DIM: int = 1536


@dataclass
class ValkeyIndexInfo:
    """The default Valkey Vector Store Index Info."""

    name: str = "llama_index"
    """The unique name of the index."""
    prefix: str = "llama_index/vector"
    """The prefix used for Valkey keys associated with this index."""
    key_separator: str = "_"
    """The separator character used in designing Valkey keys."""
    storage_type: DataType = DataType.HASH
    """The storage type used in Valkey (HASH or JSON)."""


class ValkeyVectorStoreSchema:
    """The default Valkey Vector Store Schema."""

    def __init__(self, **data: Any) -> None:
        self.index = ValkeyIndexInfo()
        # For HASH storage, use field names directly (no $ prefix)
        # For JSON storage, you would use f"$.{field_name}"
        self.fields: List[Field] = [
            TagField(NODE_ID_FIELD_NAME, NODE_ID_FIELD_NAME),
            TagField(DOC_ID_FIELD_NAME, DOC_ID_FIELD_NAME),
            TextField(TEXT_FIELD_NAME, TEXT_FIELD_NAME),
            TextField(NODE_CONTENT_FIELD_NAME, NODE_CONTENT_FIELD_NAME),
            VectorField(
                name=VECTOR_FIELD_NAME,
                algorithm=VectorAlgorithm.FLAT,
                attributes=VectorFieldAttributesFlat(
                    dimensions=DEFAULT_EMBEDDING_DIM,
                    distance_metric=DistanceMetricType.COSINE,
                    type=VectorType.FLOAT32,
                ),
                alias=VECTOR_FIELD_NAME,
            ),
        ]

        # Apply any custom data passed in
        if data:
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def add_fields(self, fields: List[Field]) -> None:
        """
        Add custom fields to the schema.

        Args:
            fields: List of Field objects (TagField, TextField, NumericField, VectorField)

        Example:
            schema = ValkeyVectorStoreSchema()
            schema.add_fields([
                TagField("category", "category"),
                NumericField("price", "price")
            ])

        """
        self.fields.extend(fields)
