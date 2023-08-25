"""Logger class."""

from typing import Any, Dict, List, Set


class LlamaLogger:
    """Logger class."""

    def __init__(self) -> None:
        """Init params."""
        self._logs: List[Dict] = []
        self._metadata: Dict[str, Any] = {}

    def reset(self) -> None:
        """Reset logs."""
        self._logs = []

    def set_metadata(self, metadata: Dict) -> None:
        """Set metadata."""
        self._metadata.update(metadata)

    def unset_metadata(self, metadata_keys: Set) -> None:
        """Unset metadata."""
        for key in metadata_keys:
            self._metadata.pop(key, None)

    def get_metadata(self) -> Dict:
        """Get metadata."""
        return self._metadata

    def add_log(self, log: Dict) -> None:
        """Add log."""
        updated_log = {**self._metadata, **log}
        # TODO: figure out better abstraction
        self._logs.append(updated_log)

    def get_logs(self) -> List[Dict]:
        """Get logs."""
        return self._logs
