"""Tests for plugin marketplace and plugin CLI handlers."""

import json
import tempfile
from io import StringIO
from unittest import mock

import pytest

from llama_index.core.llama_pack.marketplace import Marketplace, MarketplaceManager
from llama_index.cli.command_line import (
    fetch_marketplace_catalog,
    handle_marketplace_add,
    handle_marketplace_list,
    handle_marketplace_remove,
    handle_plugin_install,
    handle_plugin_list,
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory for marketplace tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def manager(temp_config_dir):
    """Create a MarketplaceManager with a temp config directory."""
    return MarketplaceManager(config_dir=temp_config_dir)


# --- MarketplaceManager unit tests ---


class TestMarketplaceDataclass:
    def test_to_dict(self):
        m = Marketplace(
            name="test",
            repository="org/repo",
            branch="dev",
            base_path="packs",
            description="A test marketplace",
        )
        d = m.to_dict()
        assert d == {
            "name": "test",
            "repository": "org/repo",
            "branch": "dev",
            "base_path": "packs",
            "description": "A test marketplace",
        }

    def test_from_dict(self):
        d = {
            "name": "test",
            "repository": "org/repo",
            "branch": "main",
            "base_path": "",
            "description": "desc",
        }
        m = Marketplace.from_dict(d)
        assert m.name == "test"
        assert m.repository == "org/repo"

    def test_defaults(self):
        m = Marketplace(name="x", repository="a/b")
        assert m.branch == "main"
        assert m.base_path == ""
        assert m.description == ""


class TestMarketplaceManager:
    def test_default_marketplaces_loaded(self, manager):
        marketplaces = manager.list_marketplaces()
        names = [m.name for m in marketplaces]
        assert "llamaindex" in names
        assert "superpowers-marketplace" in names
        assert "llama-hub" in names
        assert "llama-lab" in names

    def test_add_marketplace(self, manager):
        result = manager.add_marketplace(
            name="test-mp",
            repository="test/repo",
            branch="dev",
            description="Test",
        )
        assert result is True
        mp = manager.get_marketplace("test-mp")
        assert mp is not None
        assert mp.repository == "test/repo"
        assert mp.branch == "dev"

    def test_add_duplicate_marketplace(self, manager):
        manager.add_marketplace(name="dup", repository="a/b")
        result = manager.add_marketplace(name="dup", repository="c/d")
        assert result is False

    def test_remove_marketplace(self, manager):
        manager.add_marketplace(name="removable", repository="x/y")
        result = manager.remove_marketplace("removable")
        assert result is True
        assert manager.get_marketplace("removable") is None

    def test_remove_nonexistent_marketplace(self, manager):
        result = manager.remove_marketplace("does-not-exist")
        assert result is False

    def test_cannot_remove_default_marketplace(self, manager):
        result = manager.remove_marketplace("llamaindex")
        assert result is False
        assert manager.get_marketplace("llamaindex") is not None

    def test_get_marketplace_url(self, manager):
        url = manager.get_marketplace_url("llamaindex")
        assert url == (
            "https://raw.githubusercontent.com/run-llama/llama_index/main"
            "/llama-index-packs"
        )

    def test_get_marketplace_url_no_base_path(self, manager):
        url = manager.get_marketplace_url("superpowers-marketplace")
        assert url == (
            "https://raw.githubusercontent.com/obra/superpowers-marketplace/main"
        )

    def test_get_marketplace_url_not_found(self, manager):
        assert manager.get_marketplace_url("nope") is None

    def test_get_marketplace_source_url(self, manager):
        url = manager.get_marketplace_source_url("llamaindex")
        assert url == "https://github.com/run-llama/llama_index/tree/main"

    def test_get_marketplace_source_url_not_found(self, manager):
        assert manager.get_marketplace_source_url("nope") is None

    def test_config_persistence(self, temp_config_dir):
        m1 = MarketplaceManager(config_dir=temp_config_dir)
        m1.add_marketplace(name="persisted", repository="p/q")

        m2 = MarketplaceManager(config_dir=temp_config_dir)
        assert m2.get_marketplace("persisted") is not None

    def test_corrupted_config_falls_back_to_defaults(self, temp_config_dir):
        config_file = f"{temp_config_dir}/marketplaces.json"
        with open(config_file, "w") as f:
            f.write("NOT VALID JSON")

        manager = MarketplaceManager(config_dir=temp_config_dir)
        marketplaces = manager.list_marketplaces()
        names = [m.name for m in marketplaces]
        assert "llamaindex" in names


# --- CLI handler tests ---


class TestHandleMarketplaceAdd:
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_successful_add(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.add_marketplace.return_value = True
        mock_mgr_cls.return_value = mock_mgr

        handle_marketplace_add(
            name="new-mp",
            repository="org/repo",
            branch="main",
            base_path="",
            description="",
        )

        mock_mgr.add_marketplace.assert_called_once_with(
            name="new-mp",
            repository="org/repo",
            branch="main",
            base_path="",
            description="",
        )
        output = capsys.readouterr().out
        assert "Successfully added" in output
        assert "new-mp" in output

    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_duplicate_add(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.add_marketplace.return_value = False
        mock_mgr_cls.return_value = mock_mgr

        handle_marketplace_add(name="dup", repository="a/b")

        output = capsys.readouterr().out
        assert "already exists" in output


class TestHandleMarketplaceList:
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_list_marketplaces(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.list_marketplaces.return_value = [
            Marketplace(
                name="test-mp",
                repository="t/r",
                branch="main",
                description="Test",
            ),
        ]
        mock_mgr_cls.return_value = mock_mgr

        handle_marketplace_list()

        output = capsys.readouterr().out
        assert "test-mp" in output
        assert "t/r" in output

    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_list_empty(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.list_marketplaces.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        handle_marketplace_list()

        output = capsys.readouterr().out
        assert "No marketplaces" in output


class TestHandleMarketplaceRemove:
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_successful_remove(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.remove_marketplace.return_value = True
        mock_mgr_cls.return_value = mock_mgr

        handle_marketplace_remove(name="gone")

        output = capsys.readouterr().out
        assert "Successfully removed" in output

    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_remove_not_found(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.remove_marketplace.return_value = False
        mock_mgr_cls.return_value = mock_mgr

        handle_marketplace_remove(name="nope")

        output = capsys.readouterr().out
        assert "not found or cannot be removed" in output


class TestHandlePluginInstall:
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_missing_marketplace_specifier(self, mock_mgr_cls, capsys):
        handle_plugin_install(plugin_identifier="NoAtSign")

        output = capsys.readouterr().out
        assert "must include a marketplace specifier" in output

    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_marketplace_not_found(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.get_marketplace.return_value = None
        mock_mgr_cls.return_value = mock_mgr

        handle_plugin_install(plugin_identifier="Pack@unknown-mp")

        output = capsys.readouterr().out
        assert "not found" in output
        assert "unknown-mp" in output

    @mock.patch("llama_index.cli.command_line.download_llama_pack")
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_successful_install(self, mock_mgr_cls, mock_download, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.get_marketplace.return_value = Marketplace(
            name="superpowers-marketplace",
            repository="obra/superpowers-marketplace",
            branch="main",
        )
        mock_mgr_cls.return_value = mock_mgr
        mock_download.return_value = mock.Mock()

        handle_plugin_install(
            plugin_identifier="superpowers@superpowers-marketplace",
            download_dir="./test_dir",
        )

        mock_download.assert_called_once_with(
            llama_pack_class="superpowers@superpowers-marketplace",
            download_dir="./test_dir",
        )
        output = capsys.readouterr().out
        assert "Successfully installed" in output

    @mock.patch("llama_index.cli.command_line.download_llama_pack")
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_install_with_default_dir(self, mock_mgr_cls, mock_download, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.get_marketplace.return_value = Marketplace(
            name="mp", repository="a/b", branch="main",
        )
        mock_mgr_cls.return_value = mock_mgr
        mock_download.return_value = mock.Mock()

        handle_plugin_install(plugin_identifier="MyPack@mp")

        mock_download.assert_called_once_with(
            llama_pack_class="MyPack@mp",
            download_dir="./plugins/mypack",
        )

    @mock.patch("llama_index.cli.command_line.download_llama_pack")
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_install_handles_download_error(self, mock_mgr_cls, mock_download, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.get_marketplace.return_value = Marketplace(
            name="mp", repository="a/b", branch="main",
        )
        mock_mgr_cls.return_value = mock_mgr
        mock_download.side_effect = ValueError("Pack not found")

        handle_plugin_install(plugin_identifier="Bad@mp")

        output = capsys.readouterr().out
        assert "Error installing plugin" in output


class TestHandlePluginList:
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_list_plugins(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.list_marketplaces.return_value = [
            Marketplace(
                name="superpowers-marketplace",
                repository="obra/superpowers-marketplace",
                description="Curated plugins",
            ),
        ]
        mock_mgr_cls.return_value = mock_mgr

        handle_plugin_list()

        output = capsys.readouterr().out
        assert "superpowers-marketplace" in output
        assert "obra/superpowers-marketplace" in output
        assert "Curated plugins" in output

    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    def test_list_plugins_empty(self, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.list_marketplaces.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        handle_plugin_list()

        output = capsys.readouterr().out
        assert "No marketplaces" in output


# --- Live catalog discovery tests ---


class TestHandlePluginListWithCatalog:
    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    @mock.patch("llama_index.cli.command_line.fetch_marketplace_catalog")
    def test_list_shows_live_catalog(self, mock_fetch, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.list_marketplaces.return_value = [
            Marketplace(
                name="superpowers-marketplace",
                repository="obra/superpowers-marketplace",
                description="Curated plugins",
            ),
        ]
        mock_mgr_cls.return_value = mock_mgr
        mock_fetch.return_value = {
            "plugins": [
                {
                    "name": "superpowers",
                    "description": "Core skills library",
                    "version": "4.3.0",
                },
                {
                    "name": "episodic-memory",
                    "description": "Semantic search for conversations",
                    "version": "1.0.15",
                },
            ]
        }

        handle_plugin_list()

        output = capsys.readouterr().out
        assert "superpowers" in output
        assert "episodic-memory" in output
        assert "4.3.0" in output

    @mock.patch("llama_index.cli.command_line.MarketplaceManager")
    @mock.patch("llama_index.cli.command_line.fetch_marketplace_catalog")
    def test_list_graceful_on_catalog_failure(self, mock_fetch, mock_mgr_cls, capsys):
        mock_mgr = mock.Mock()
        mock_mgr.list_marketplaces.return_value = [
            Marketplace(
                name="superpowers-marketplace",
                repository="obra/superpowers-marketplace",
                description="Curated plugins",
            ),
        ]
        mock_mgr_cls.return_value = mock_mgr
        mock_fetch.return_value = None

        handle_plugin_list()

        output = capsys.readouterr().out
        assert "superpowers-marketplace" in output
        # Should still show marketplace info even when catalog fetch fails
