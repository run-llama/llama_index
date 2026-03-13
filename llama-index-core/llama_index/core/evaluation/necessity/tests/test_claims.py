from llama_index.core.evaluation.necessity.claims import extract_claims


def test_extract_claims_basic():
    answer = "A. B! C?"
    claims = extract_claims(answer)
    assert claims == ["A.", "B!", "C?"]


def test_extract_claims_empty():
    assert extract_claims("") == []
