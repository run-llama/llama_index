"""
Regression tests for CWE-470 (unsafe reflection) fix in
``llama_index.ingestion.ray.utils``.

Security regression tests (these fail without the fix) cover the two
independent guards added to ``_safe_load_node_class`` — the module
namespace allowlist, including its trailing-dot boundary — the
``BaseNode`` subclass check, and confirmation that the real call path
(``ray_deserialize_node``) actually exercises the guarded helper rather
than being a dead/unused fix.

Two of the tests -- ``test_safe_load_node_class_allows_legit_basenode_subclass``
and ``test_ray_serialize_deserialize_round_trip`` -- are compatibility
tests rather than security regression proof: they pass both with and
without the fix, and exist to show the guards do not break legitimate
(de)serialization.
"""

import pytest
from llama_index.core.schema import BaseNode, TextNode

from llama_index.ingestion.ray.utils import (
    _safe_load_node_class,
    ray_deserialize_node,
    ray_serialize_node,
)


def test_safe_load_node_class_rejects_untrusted_module() -> None:
    """
    PoC attack path: a serialized node claiming module="os",
    class_name="system" must never resolve to os.system. Before the
    fix, _safe_load_node_class would happily import "os" and return
    the "system" attribute, which combined with cls.from_json(...)
    in ray_deserialize_node is an arbitrary-code-execution primitive.
    """
    with pytest.raises(ValueError, match="Untrusted module"):
        _safe_load_node_class("os", "system")


def test_safe_load_node_class_rejects_non_basenode_in_namespace() -> None:
    """
    Even inside the llama_index.* namespace, resolving a class that is
    not a BaseNode subclass must be rejected. This proves the second
    guard (isinstance/issubclass check) is doing real work and the fix
    isn't relying solely on the module-prefix allowlist -- an attacker
    who can influence class_name (but not module) shouldn't be able to
    smuggle out an arbitrary llama_index.* class either.
    """
    # MetadataMode is a real, importable Enum inside the allowed namespace.
    with pytest.raises(ValueError, match="not a BaseNode subclass"):
        _safe_load_node_class("llama_index.core.schema", "MetadataMode")


def test_safe_load_node_class_rejects_lookalike_namespace() -> None:
    """
    The allowlist prefix is "llama_index." *with* the trailing dot, so a
    third-party package whose name merely starts with the same letters --
    e.g. an attacker-published "llama_index_evil" on PyPI -- must not be
    treated as trusted. This test locks the boundary down: dropping the
    trailing dot from _ALLOWED_MODULE_PREFIXES would silently reopen the
    hole while every other test here still passed.
    """
    with pytest.raises(ValueError, match="Untrusted module"):
        _safe_load_node_class("llama_index_evil", "TextNode")


def test_safe_load_node_class_allows_legit_basenode_subclass() -> None:
    """
    The legitimate case: a real BaseNode subclass inside the
    llama_index namespace must resolve normally so the fix doesn't
    break ordinary (de)serialization.
    """
    cls = _safe_load_node_class("llama_index.core.schema", "TextNode")
    assert cls is TextNode
    assert issubclass(cls, BaseNode)


def test_ray_deserialize_node_routes_through_safe_loader() -> None:
    """
    ray_deserialize_node must actually call _safe_load_node_class on
    the real path -- not just have the guarded helper sitting unused
    elsewhere in the module. Feed it a serialized_node whose "module"
    is the same "os" PoC payload used above and confirm it raises the
    same ValueError instead of reaching cls.from_json(...).
    """
    malicious_payload = {
        "module": "os",
        "class_name": "system",
        "data": "{}",
        "embedding": None,
    }
    with pytest.raises(ValueError, match="Untrusted module"):
        ray_deserialize_node(malicious_payload)


def test_ray_serialize_deserialize_round_trip() -> None:
    """
    Compatibility check that legitimate traffic through the guarded path
    is unaffected: serialize a real TextNode and deserialize it back to an
    equivalent node. Deliberately asserts on behaviour rather than on the
    serialized module/class strings, which are implementation details that
    would make this test fragile across llama_index versions.
    """
    original = TextNode(text="hello world", id_="node-1")
    serialized = ray_serialize_node(original)

    restored = ray_deserialize_node(serialized)

    assert isinstance(restored, TextNode)
    assert restored.id_ == original.id_
    assert restored.text == original.text
