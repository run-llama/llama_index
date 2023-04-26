import socket

import pytest


@pytest.fixture(autouse=True)
def no_networking(monkeypatch: pytest.MonkeyPatch):
    def deny_network(*args, **kwargs):
        raise RuntimeError("Network access denied for test")

    monkeypatch.setattr(socket, "socket", deny_network)


@pytest.fixture
def allow_networking(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.undo()
