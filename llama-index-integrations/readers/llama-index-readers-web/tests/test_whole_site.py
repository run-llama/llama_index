from llama_index.readers.web import WholeSiteReader


class FakeDriver:
    """Stands in for the real Selenium webdriver.Chrome, driven by a fixed link graph."""

    def __init__(self, graph):
        self.graph = graph
        self.current = None

    def get(self, url):
        self.current = url

    def quit(self):
        pass


def _make_reader(prefix: str, graph: dict) -> WholeSiteReader:
    reader = WholeSiteReader(prefix=prefix, driver=FakeDriver(graph))
    reader.extract_content = lambda: reader.driver.graph[reader.driver.current][0]
    reader.extract_links = lambda: reader.driver.graph[reader.driver.current][1]
    return reader


def test_look_alike_domain_is_not_crawled():
    """
    A prefix of https://example.com must not match https://example.com.evil.com,
    since that is a different origin, not a substring match.
    """
    prefix = "https://example.com"
    lookalike = "https://example.com.evil.com/harvest"
    graph = {
        f"{prefix}/home": ("home", [lookalike]),
        lookalike: ("attacker page", []),
    }

    docs = _make_reader(prefix, graph).load_data(f"{prefix}/home")
    visited = {d.extra_info["URL"] for d in docs}

    assert visited == {f"{prefix}/home"}


def test_same_origin_links_are_still_crawled():
    """Normal same-origin crawling behavior must be unaffected by the scope fix."""
    prefix = "https://example.com/"
    start = "https://example.com/index"
    graph = {
        start: ("index", ["https://example.com/about"]),
        "https://example.com/about": ("about", []),
    }

    docs = _make_reader(prefix, graph).load_data(start)
    visited = {d.extra_info["URL"] for d in docs}

    assert visited == set(graph.keys())
