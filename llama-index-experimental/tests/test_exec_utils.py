from llama_index.experimental.exec_utils import _contains_protected_access


def test_contains_protected_access() -> None:
    assert not _contains_protected_access("def _a(b): pass"), (
        "definition of dunder function"
    )
    assert _contains_protected_access("a = _b(c)"), "call to protected function"
    assert not _contains_protected_access("a = b(c)"), "call to public function"
    assert _contains_protected_access("_b"), "access to protected name"
    assert not _contains_protected_access("b"), "access to public name"
    assert _contains_protected_access("_b[0]"), "subscript access to protected name"
    assert not _contains_protected_access("b[0]"), "subscript access to public name"
    assert _contains_protected_access("_a.b"), "access to attribute of a protected name"
    assert not _contains_protected_access("a.b"), "access to attribute of a public name"
    assert _contains_protected_access("a._b"), "access to protected attribute of a name"
    assert not _contains_protected_access("a.b"), "access to public attribute of a name"
