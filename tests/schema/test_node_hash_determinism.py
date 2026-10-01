import hashlib

def compute_node_hash(text: str, metadata: dict) -> str:
    sorted_meta = sorted(metadata.items())
    payload = f"{text}|{sorted_meta}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def test_node_hash_deterministic():
    h1 = compute_node_hash("sample document", {"page": 1, "source": "pdf"})
    h2 = compute_node_hash("sample document", {"source": "pdf", "page": 1})
    assert h1 == h2

def test_distinct_text_hashes():
    h1 = compute_node_hash("", {})
    h2 = compute_node_hash("   ", {})
    assert h1 != h2
