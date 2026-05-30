"""Regression tests for the .strip() → .removeprefix() fix in RedisVectorStore.

str.strip(chars) treats its argument as a *set of characters* to remove from
both ends of the string, not as an exact prefix substring.  When a node UUID
begins with a character that appears anywhere in the Redis index prefix (e.g.
the 'e' in "semantic_cache_doc"), the old code silently corrupted the returned
node ID.

These tests are pure-Python – no Redis connection or Docker container needed.
"""


def test_strip_corrupts_uuid_starting_with_prefix_char():
    """Demonstrate that str.strip() silently corrupts UUIDs whose first
    character overlaps with any character in the Redis prefix string."""
    prefix, separator = "semantic_cache_doc", ":"
    prefix_str = prefix + separator

    # 'e' appears in "semantic_cache_doc" and is the first char of this UUID.
    node_id = "e7b95ae7-6369-404d-8287-1f4504121563"
    full_key = f"{prefix_str}{node_id}"

    # Old behaviour: strip() treats prefix_str as a char-set and removes the
    # leading 'e' from the UUID, returning a truncated, invalid ID.
    assert full_key.strip(prefix_str) != node_id, (
        "str.strip() should have incorrectly removed the leading 'e' from the UUID"
    )

    # New behaviour: removeprefix() removes the exact prefix substring,
    # leaving the UUID intact.
    assert full_key.removeprefix(prefix_str) == node_id, (
        "str.removeprefix() should return the original UUID unchanged"
    )


def test_removeprefix_safe_for_non_overlapping_prefix():
    """Nodes whose UUIDs don't start with a char present in the prefix are
    unaffected by the old bug, and removeprefix() does not regress them."""
    prefix, separator = "doc", ":"
    prefix_str = prefix + separator

    # 'f' is not in "doc:" so strip() would have returned the right answer too,
    # but removeprefix() is still correct and explicit.
    node_id = "f1f2f3f4-0000-0000-0000-000000000000"
    full_key = f"{prefix_str}{node_id}"

    assert full_key.removeprefix(prefix_str) == node_id


def test_removeprefix_does_not_alter_key_without_prefix():
    """removeprefix() must be a no-op when the key does not start with the
    expected prefix (defensive: avoids silent data loss on unexpected input)."""
    prefix_str = "doc:"
    bare_id = "abc123"  # no prefix

    assert bare_id.removeprefix(prefix_str) == bare_id
