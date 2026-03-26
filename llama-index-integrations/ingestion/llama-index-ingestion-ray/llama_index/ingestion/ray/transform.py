from typing import Type, Any, Dict, Optional

import ray

from llama_index.core.schema import TransformComponent
from pydantic import BaseModel
import pyarrow as pa

from llama_index.ingestion.ray.utils import (
    ray_deserialize_node_batch,
    ray_serialize_node_batch,
)


class TransformActor:
    """A Ray Actor executing the wrapped transformation."""

    def __init__(
        self,
        transform_class: Type[TransformComponent],
        transform_kwargs: Dict[str, Any],
    ):
        self.transform = transform_class(**transform_kwargs)

    def __call__(self, batch: pa.Table, **kwargs) -> pa.Table:
        """Execute the wrapped transformation on the given batch."""
        # Deserialize input nodes
        nodes = ray_deserialize_node_batch(batch)

        # Apply the transformation
        new_nodes = self.transform(nodes, **kwargs)

        # Serialize output nodes
        return ray_serialize_node_batch(new_nodes)


class RayTransformComponent(BaseModel):
    """
    A wrapper around transformations that enables execution in Ray.

    Args:
        transform_class (Type[TransformComponent]): The transformation class to wrap.
        transform_kwargs (Optional[Dict[str, Any]], optional): The keyword arguments to pass to the transformation __init__ function.
        map_batches_kwargs (Optional[Dict[str, Any]], optional): The keyword arguments to pass to ray.data.Dataset.map_batches (see https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html for details)

    """

    transform_class: Type[TransformComponent]
    transform_kwargs: Dict[str, Any]
    map_batches_kwargs: Dict[str, Any]

    def __init__(
        self,
        transform_class: Type[TransformComponent],
        map_batches_kwargs: Optional[Dict[str, Any]] = None,
        transform_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            transform_class=transform_class,
            transform_kwargs=transform_kwargs or kwargs,
            map_batches_kwargs=map_batches_kwargs or {},
        )

    def __call__(self, dataset: ray.data.Dataset, **kwargs) -> ray.data.Dataset:
        """Run the transformation on the given ray dataset."""
        return dataset.map_batches(
            TransformActor,
            fn_constructor_kwargs={
                "transform_class": self.transform_class,
                "transform_kwargs": self.transform_kwargs,
            },
            fn_kwargs=kwargs,
            batch_format="pyarrow",
            **self.map_batches_kwargs,
        )
