import base64
import json
import pickle
from typing import Any

from workflows.context.serializers import (
    BaseSerializer,
    JsonSerializer,
)


class PickleSerializer(JsonSerializer):
    """
    Hybrid serializer: JSON when possible, Pickle as a safe fallback.

    This serializer attempts JSON first for readability and portability, and
    transparently falls back to Pickle for objects that cannot be represented in
    JSON. Deserialization prioritizes JSON and falls back to Pickle.

    Warning:
        Pickle can execute arbitrary code during deserialization. Only
        deserialize trusted payloads.

    Note: Used to be called `JsonPickleSerializer` but it was renamed to `PickleSerializer`.

    """

    def serialize(self, value: Any) -> str:
        """
        Serialize with JSON preference and Pickle fallback.

        Args:
            value (Any): The value to encode.

        Returns:
            str: Encoded string (JSON or base64-encoded Pickle bytes).

        """
        try:
            return super().serialize(value)
        except Exception:
            return base64.b64encode(pickle.dumps(value)).decode("utf-8")

    def deserialize(self, value: str) -> Any:
        """
        Deserialize with JSON preference and Pickle fallback.

        Args:
            value (str): Encoded string.

        Returns:
            Any: The reconstructed value.

        Notes:
            Use only with trusted payloads due to Pickle security implications.

        """
        # Pickle payloads are base64 of bytes starting with 0x80 (protocol 2+),
        # so they never parse as valid JSON. b64decode drops non-alphabet
        # characters when validate=False, allowing JSON strings (like '"SwEu"')
        # to decode into valid pickle bytecode (e.g. b'K\x01.' -> 1).
        # Trying JSON first ensures JSON values never reach pickle.loads.
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return pickle.loads(base64.b64decode(value))
        return self.deserialize_value(data)


# provided for backward compatibility
JsonPickleSerializer = PickleSerializer

__all__ = [
    "BaseSerializer",
    "JsonPickleSerializer",
    "JsonSerializer",
    "PickleSerializer",
]
