import pytest

from llama_index.readers.web import ReadabilityWebPageReader
from llama_index.readers.web.readability_web.base import proxy_settings


@pytest.mark.parametrize(
    ("proxy", "expected"),
    [
        ("http://proxy.example.com:8080", {"server": "http://proxy.example.com:8080"}),
        ("proxy.example.com:8080", {"server": "proxy.example.com:8080"}),
        (
            "http://user:pass@proxy.example.com:8080",
            {
                "server": "http://proxy.example.com:8080",
                "username": "user",
                "password": "pass",
            },
        ),
        (
            "http://user-country-us:p%23ss%2Fw%3Fr%40d@proxy.example.com:8080",
            {
                "server": "http://proxy.example.com:8080",
                "username": "user-country-us",
                "password": "p#ss/w?r@d",
            },
        ),
        (
            "user:pass@proxy.example.com:8080",
            {
                "server": "http://proxy.example.com:8080",
                "username": "user",
                "password": "pass",
            },
        ),
        (
            "socks5://user@proxy.example.com:1080",
            {"server": "socks5://proxy.example.com:1080", "username": "user"},
        ),
    ],
)
def test_proxy_settings(proxy, expected):
    assert proxy_settings(proxy) == expected


def test_reader_passes_credentials_separately():
    reader = ReadabilityWebPageReader(proxy="http://user:pass@proxy.example.com:8080")
    assert reader._launch_options["proxy"] == {
        "server": "http://proxy.example.com:8080",
        "username": "user",
        "password": "pass",
    }


def test_reader_without_proxy():
    assert "proxy" not in ReadabilityWebPageReader()._launch_options
