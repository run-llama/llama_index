"""Plugin Marketplace Management System."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Marketplace:
    """Represents a plugin marketplace configuration."""

    name: str
    repository: str
    branch: str = "main"
    base_path: str = ""
    description: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Marketplace":
        """Create from dictionary."""
        return cls(**data)


class MarketplaceManager:
    """Manages plugin marketplace registrations."""

    DEFAULT_MARKETPLACES = [
        {
            "name": "llamaindex",
            "repository": "run-llama/llama_index",
            "branch": "main",
            "base_path": "llama-index-packs",
            "description": "Official LlamaIndex packs repository"
        },
        {
            "name": "superpowers-marketplace",
            "repository": "obra/superpowers-marketplace",
            "branch": "main",
            "base_path": "",
            "description": "Curated Claude Code plugin marketplace by obra"
        }
    ]

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the marketplace manager.

        Args:
            config_dir: Directory to store marketplace configuration.
                       Defaults to ~/.llamaindex/
        """
        if config_dir is None:
            config_dir = os.path.join(Path.home(), ".llamaindex")

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "marketplaces.json"

        # Initialize with default marketplaces if file doesn't exist
        if not self.config_file.exists():
            self._save_marketplaces(
                [Marketplace.from_dict(m) for m in self.DEFAULT_MARKETPLACES]
            )

    def _load_marketplaces(self) -> List[Marketplace]:
        """Load marketplaces from config file."""
        if not self.config_file.exists():
            return [Marketplace.from_dict(m) for m in self.DEFAULT_MARKETPLACES]

        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                return [Marketplace.from_dict(m) for m in data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading marketplace config: {e}. Using defaults.")
            return [Marketplace.from_dict(m) for m in self.DEFAULT_MARKETPLACES]

    def _save_marketplaces(self, marketplaces: List[Marketplace]) -> None:
        """Save marketplaces to config file."""
        with open(self.config_file, 'w') as f:
            json.dump([m.to_dict() for m in marketplaces], f, indent=2)

    def add_marketplace(
        self,
        name: str,
        repository: str,
        branch: str = "main",
        base_path: str = "",
        description: str = ""
    ) -> bool:
        """
        Add a new marketplace.

        Args:
            name: Short name for the marketplace (used in install commands)
            repository: GitHub repository in format 'owner/repo'
            branch: Git branch to use (default: 'main')
            base_path: Base path within the repository for packs
            description: Human-readable description

        Returns:
            True if added successfully, False if marketplace already exists
        """
        marketplaces = self._load_marketplaces()

        # Check if marketplace already exists
        if any(m.name == name for m in marketplaces):
            return False

        marketplace = Marketplace(
            name=name,
            repository=repository,
            branch=branch,
            base_path=base_path,
            description=description
        )

        marketplaces.append(marketplace)
        self._save_marketplaces(marketplaces)
        return True

    def remove_marketplace(self, name: str) -> bool:
        """
        Remove a marketplace.

        Args:
            name: Name of the marketplace to remove

        Returns:
            True if removed successfully, False if not found
        """
        marketplaces = self._load_marketplaces()

        # Don't allow removing the default llamaindex marketplace
        if name == "llamaindex":
            logger.warning("Cannot remove the default 'llamaindex' marketplace")
            return False

        original_count = len(marketplaces)
        marketplaces = [m for m in marketplaces if m.name != name]

        if len(marketplaces) == original_count:
            return False

        self._save_marketplaces(marketplaces)
        return True

    def list_marketplaces(self) -> List[Marketplace]:
        """
        List all registered marketplaces.

        Returns:
            List of Marketplace objects
        """
        return self._load_marketplaces()

    def get_marketplace(self, name: str) -> Optional[Marketplace]:
        """
        Get a marketplace by name.

        Args:
            name: Name of the marketplace

        Returns:
            Marketplace object or None if not found
        """
        marketplaces = self._load_marketplaces()
        for marketplace in marketplaces:
            if marketplace.name == name:
                return marketplace
        return None

    def get_marketplace_url(self, name: str) -> Optional[str]:
        """
        Get the raw content URL for a marketplace.

        Args:
            name: Name of the marketplace

        Returns:
            URL string or None if marketplace not found
        """
        marketplace = self.get_marketplace(name)
        if marketplace is None:
            return None

        # Construct raw GitHub URL
        url = f"https://raw.githubusercontent.com/{marketplace.repository}/{marketplace.branch}"
        if marketplace.base_path:
            url = f"{url}/{marketplace.base_path}"

        return url

    def get_marketplace_source_url(self, name: str) -> Optional[str]:
        """
        Get the GitHub tree URL for a marketplace (for browsing source files).

        Args:
            name: Name of the marketplace

        Returns:
            URL string or None if marketplace not found
        """
        marketplace = self.get_marketplace(name)
        if marketplace is None:
            return None

        return f"https://github.com/{marketplace.repository}/tree/{marketplace.branch}"
