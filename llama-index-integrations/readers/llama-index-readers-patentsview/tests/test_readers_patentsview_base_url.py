from llama_index.readers.patentsview import PatentsviewReader


def test_default_url(monkeypatch):
    monkeypatch.delenv("PATENTSVIEW_BASE_URL", raising=False)
    r = PatentsviewReader(api_key="x")
    assert r.url == "https://search.patentsview.org/api/v1/patent"


def test_env_url(monkeypatch):
    monkeypatch.setenv("PATENTSVIEW_BASE_URL", "https://example.org/")
    r = PatentsviewReader(api_key="x")
    assert r.url == "https://example.org/api/v1/patent"


def test_arg_beats_env(monkeypatch):
    monkeypatch.setenv("PATENTSVIEW_BASE_URL", "https://env.example.org")
    r = PatentsviewReader(api_key="x", base_url="https://arg.example.org")
    assert r.url == "https://arg.example.org/api/v1/patent"
