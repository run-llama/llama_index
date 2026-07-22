import base64
import httpx
import logging
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest
from llama_index.core.schema import (
    Document,
    ImageDocument,
    ImageNode,
    MediaResource,
    MetadataMode,
    NodeWithScore,
    ObjectType,
    TextNode,
    _ssrf_redirect_hook,
    _ssrf_safe_get,
    _validate_ssrf_url,
)


@pytest.fixture()
def text_node() -> TextNode:
    return TextNode(
        text="hello world",
        metadata={"foo": "bar"},
        embedding=[0.1, 0.2, 0.3],
    )


@pytest.fixture()
def node_with_score(text_node: TextNode) -> NodeWithScore:
    return NodeWithScore(
        node=text_node,
        score=0.5,
    )


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def png_1px(png_1px_b64) -> bytes:
    return base64.b64decode(png_1px_b64)


def test_node_with_score_passthrough(node_with_score: NodeWithScore) -> None:
    _ = node_with_score.id_
    _ = node_with_score.node_id
    _ = node_with_score.text
    _ = node_with_score.metadata
    _ = node_with_score.embedding
    _ = node_with_score.get_text()
    _ = node_with_score.get_content()
    _ = node_with_score.get_embedding()


def test_text_node_hash() -> None:
    node = TextNode(text="hello", metadata={"foo": "bar"})
    assert (
        node.hash == "aa158bf3388f103cef4bd85b2ca93f343ad8f5e50f58ae4141a35d75a2f21fb0"
    )
    node.set_content("world")
    assert (
        node.hash == "ce6a3cefc3451ecb1ff41ec41a7d7e24354983520d8b2d6f5447be0b6b9b6b99"
    )

    node.text = "new"
    assert (
        node.hash == "bef8ff82498c9aa7d9f9751f441da9a1a1c4e9941bd03c57caa4a602cd5cadd0"
    )
    node2 = TextNode(text="new", metadata={"foo": "bar"})
    assert node2.hash == node.hash
    node3 = TextNode(text="new", metadata={"foo": "baz"})
    assert node3.hash != node.hash


def test_text_node_with_text_resource():
    tr = MediaResource(text="This is a test")
    text_node = TextNode(text_resource=tr)
    assert text_node.text == "This is a test"

    tr_dict = tr.model_dump()
    text_node = TextNode(text_resource=tr_dict)
    assert text_node.text == "This is a test"


def test_text_node_metadata_separator_consistency() -> None:
    """
    Test that metadata_separator works consistently on TextNode.

    Regression test for https://github.com/run-llama/llama_index/issues/20645.
    Previously, TextNode defined its own `metadata_seperator` field that
    shadowed the parent BaseNode's `metadata_separator` (with alias
    `metadata_seperator`), causing the correctly-spelled kwarg to be ignored
    when generating content.
    """
    metadata = {"name": "Foo", "uuid": "Bar"}

    # Using the correctly-spelled kwarg should work on TextNode
    node_correct = TextNode(
        text="content",
        metadata=metadata,
        metadata_separator="::SEP::",
    )
    # Using the alias (typo) kwarg should also work
    node_alias = TextNode(
        text="content",
        metadata=metadata,
        metadata_seperator="::SEP::",
    )

    content_correct = node_correct.get_content(MetadataMode.LLM)
    content_alias = node_alias.get_content(MetadataMode.LLM)

    # Both should produce identical output with the custom separator
    assert "::SEP::" in content_correct, (
        f"metadata_separator='::SEP::' was ignored: {content_correct!r}"
    )
    assert content_correct == content_alias

    # Verify the separator attribute is accessible and consistent
    assert node_correct.metadata_separator == "::SEP::"
    assert node_alias.metadata_separator == "::SEP::"

    # Also verify Document and TextNode behave the same way
    doc = Document(
        text="content",
        metadata=metadata,
        metadata_separator="::SEP::",
    )
    doc_content = doc.get_content(MetadataMode.LLM)
    assert content_correct == doc_content


def test_text_node_metadata_separator_roundtrip() -> None:
    """Test that TextNode with custom metadata_separator survives serialization roundtrip."""
    node = TextNode(
        text="hello",
        metadata={"key": "value"},
        metadata_separator=" | ",
    )
    assert node.metadata_separator == " | "

    # Roundtrip through model_dump
    dumped = node.model_dump()
    restored = TextNode(**dumped)
    assert restored.metadata_separator == " | "
    assert restored.get_content(MetadataMode.LLM) == node.get_content(MetadataMode.LLM)

    # Roundtrip through JSON
    json_str = node.model_dump_json()
    restored_json = TextNode.model_validate_json(json_str)
    assert restored_json.metadata_separator == " | "


def test_image_node_hash() -> None:
    """Test hash for ImageNode."""
    node = ImageNode(image="base64", image_path="path")
    node2 = ImageNode(image="base64", image_path="path2")
    assert node.hash != node2.hash

    # id's don't count as part of the hash
    node3 = ImageNode(image_url="base64", id_="id")
    node4 = ImageNode(image_url="base64", id_="id2")
    assert node3.hash == node4.hash


def test_image_node_mimetype() -> None:
    node = ImageNode(image_path="path")
    node2 = ImageNode(image_path="path.png")

    assert node.image_mimetype is None
    assert node2.image_mimetype == "image/png"


def test_build_image_node_image_resource() -> None:
    ir = MediaResource(path="my-image.jpg", mimetype=None)
    tr = MediaResource(text="test data")
    node = ImageNode(id_="test_node", image_resource=ir, text_resource=tr)
    assert node.text == "test data"
    assert node.image_mimetype == "image/jpeg"
    assert node.image_path == "my-image.jpg"


def test_build_text_node_text_resource() -> None:
    node = TextNode(id_="test_node", text_resource=MediaResource(text="test data"))
    assert node.text == "test data"


def test_document_init(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        # Legacy init
        doc = Document(doc_id="test")
        assert doc.doc_id == "test"
        assert doc.id_ == "test"
        # Legacy init mixed with new
        doc = Document(id_="test", doc_id="legacy_test")
        assert "'doc_id' is deprecated and 'id_' will be used instead" in caplog.text
        assert doc.id_ == "test"
        caplog.clear()

        # Legacy init
        doc = Document(extra_info={"key": "value"})
        assert doc.metadata == {"key": "value"}
        assert doc.extra_info == {"key": "value"}
        # Legacy init mixed with new
        doc = Document(extra_info={"old_key": "old_value"}, metadata={"key": "value"})
        assert (
            "'extra_info' is deprecated and 'metadata' will be used instead"
            in caplog.text
        )
        assert doc.metadata == {"key": "value"}
        assert doc.extra_info == {"key": "value"}
        caplog.clear()

        # Legacy init
        doc = Document(text="test")
        assert doc.text == "test"
        assert doc.text_resource
        assert doc.text_resource.text == "test"
        # Legacy init mixed with new
        doc = Document(text="legacy_test", text_resource=MediaResource(text="test"))
        assert (
            "'text' is deprecated and 'text_resource' will be used instead"
            in caplog.text
        )
        assert doc.text == "test"
        assert doc.text_resource
        assert doc.text_resource.text == "test"


def test_document_properties():
    doc = Document()
    assert doc.get_type() == ObjectType.DOCUMENT
    doc.doc_id = "test"
    assert doc.id_ == "test"


def test_document_str():
    with mock.patch("llama_index.core.schema.TRUNCATE_LENGTH", 5):
        doc = Document(
            id_="test_id",
            text="Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        )
        assert str(doc) == "Doc ID: test_id\nText: Lo..."


def test_document_legacy_roundtrip():
    origin = Document(id_="test_id", text="this is a test")
    assert origin.model_dump() == {
        "id_": "test_id",
        "embedding": None,
        "metadata": {},
        "excluded_embed_metadata_keys": [],
        "excluded_llm_metadata_keys": [],
        "relationships": {},
        "metadata_template": "{key}: {value}",
        "metadata_separator": "\n",
        "text": "this is a test",
        "text_resource": {
            "embeddings": None,
            "text": "this is a test",
            "mimetype": None,
            "path": None,
            "url": None,
        },
        "image_resource": None,
        "audio_resource": None,
        "video_resource": None,
        "text_template": "{metadata_str}\n\n{content}",
        "class_name": "Document",
    }
    dest = Document(**origin.model_dump())
    assert dest.text == "this is a test"


def test_document_model_dump_exclude():
    doc = Document(id_="test_id", text="this is a test")
    model_dump = doc.model_dump(exclude={"text", "metadata", "relationships"})
    assert "text" not in model_dump
    assert "metadata" not in model_dump
    assert "relationships" not in model_dump
    assert model_dump == {
        "id_": "test_id",
        "embedding": None,
        "excluded_embed_metadata_keys": [],
        "excluded_llm_metadata_keys": [],
        "metadata_template": "{key}: {value}",
        "metadata_separator": "\n",
        "text_resource": {
            "embeddings": None,
            "text": "this is a test",
            "mimetype": None,
            "path": None,
            "url": None,
        },
        "image_resource": None,
        "audio_resource": None,
        "video_resource": None,
        "text_template": "{metadata_str}\n\n{content}",
        "class_name": "Document",
    }


def test_image_document_empty():
    doc = ImageDocument(id_="test")
    assert doc.id_ == "test"
    assert doc.image is None
    assert doc.image_path is None
    assert doc.image_url is None
    assert doc.image_mimetype is None
    assert doc.class_name() == "ImageDocument"


def test_image_document_image():
    doc = ImageDocument(id_="test", image=b"123456")
    assert doc.image == "MTIzNDU2"
    doc.image = "123456789"
    assert doc.image == "MTIzNDU2Nzg5"


def test_image_document_path(tmp_path: Path):
    content = httpx.get(
        "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"
    ).content
    fl_path = tmp_path / "test_image.png"
    fl_path.write_bytes(content)
    doc = ImageDocument(id_="test", image_path=fl_path)
    assert doc.image_path == str(fl_path)
    doc.image_path = str(fl_path.parent)
    assert doc.image_path == str(fl_path.parent)


def test_image_document_url():
    doc = ImageDocument(
        id_="test",
        image_url="https://astrabert.github.io/hophop-science/images/whale_doing_science.png",
    )
    assert (
        doc.image_url
        == "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"
    )
    doc.image_url = "https://foo.org"
    assert doc.image_url == "https://foo.org/"


def test_image_document_mimetype():
    doc = ImageDocument(image=b"123456")
    assert doc.image_mimetype is None
    doc.image_mimetype = "foo"
    assert doc.image_mimetype == "foo"


def test_image_document_embeddings():
    doc = ImageDocument(text="foo")
    assert doc.text_resource is not None
    assert doc.text_embedding is None
    doc.text_embedding = [1.0, 2.0, 3.0]
    assert doc.text_embedding == [1.0, 2.0, 3.0]
    assert doc.text_resource.embeddings == {"dense": [1.0, 2.0, 3.0]}


def test_image_document_path_serialization(tmp_path: Path):
    content = httpx.get(
        "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"
    ).content
    fl_path = tmp_path / "test_image.png"
    fl_path.write_bytes(content)
    doc = ImageDocument(image_path=fl_path)
    assert doc.model_dump()["image_resource"]["path"] == fl_path.__str__()

    new_doc = ImageDocument(**doc.model_dump())
    assert new_doc.image_resource.path == fl_path


def test_image_block_resolve_image(png_1px: bytes, png_1px_b64: bytes):
    doc = ImageDocument()
    assert doc.resolve_image().read() == b""

    doc = ImageDocument(image=png_1px)

    img = doc.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = doc.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64


def test_media_resource_hash_distinguishes_empty_string_from_none() -> None:
    """Test that MediaResource.hash distinguishes between text='' and text=None."""
    resource_empty_text = MediaResource(text="")
    resource_none_text = MediaResource(text=None)

    # Both should have different hashes
    assert resource_empty_text.hash != resource_none_text.hash

    # None should return empty string (no content)
    assert resource_none_text.hash == ""

    # Empty string should return a valid hash (empty string is valid content)
    assert resource_empty_text.hash != ""
    assert len(resource_empty_text.hash) == 64  # SHA256 hex digest length


def test_media_resource_hash_with_various_content() -> None:
    """Test MediaResource.hash with different content types."""
    # Test with text content
    resource_text = MediaResource(text="hello")
    assert resource_text.hash != ""
    assert len(resource_text.hash) == 64

    # Test with path
    resource_path = MediaResource(path=Path("/tmp/test.txt"))
    assert resource_path.hash != ""
    assert len(resource_path.hash) == 64

    # Test with url
    resource_url = MediaResource(url="https://example.com")
    assert resource_url.hash != ""
    assert len(resource_url.hash) == 64

    # All different content should produce different hashes
    assert resource_text.hash != resource_path.hash
    assert resource_text.hash != resource_url.hash
    assert resource_path.hash != resource_url.hash

    # Same content should produce same hash
    resource_text2 = MediaResource(text="hello")
    assert resource_text.hash == resource_text2.hash


# --- SSRF protection (CWE-918) -----------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",  # cloud metadata
        "http://127.0.0.1/",  # loopback
        "http://10.0.0.1/",  # RFC-1918 private
        "http://172.16.0.5/",  # RFC-1918 private
        "http://192.168.1.1/",  # RFC-1918 private
        "http://0.0.0.0/",  # unspecified
        "http://[::1]/",  # IPv6 loopback
        "http://2130706433/",  # decimal-obfuscated 127.0.0.1
        "http://0177.0.0.1/",  # octal-obfuscated 127.0.0.1
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped IPv6 loopback
        "http://[::ffff:169.254.169.254]/",  # IPv4-mapped IPv6 metadata IP
    ],
)
def test_validate_ssrf_url_blocks_private_and_reserved(url: str) -> None:
    with pytest.raises(ValueError):
        _validate_ssrf_url(url)


def test_validate_ssrf_url_allows_public_host() -> None:
    # example.com resolves to a public IP; should not raise.
    _validate_ssrf_url("https://example.com/image.png")


@pytest.mark.parametrize("scheme_url", ["file:///etc/passwd", "ftp://example.com/x"])
def test_validate_ssrf_url_rejects_non_http_scheme(scheme_url: str) -> None:
    with pytest.raises(ValueError):
        _validate_ssrf_url(scheme_url)


def test_ssrf_redirect_hook_blocks_redirect_to_private_ip() -> None:
    response = mock.MagicMock()
    response.is_redirect = True
    response.url = "https://example.com/redirector"
    response.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

    with pytest.raises(ValueError):
        _ssrf_redirect_hook(response)


def test_ssrf_redirect_hook_resolves_relative_redirect() -> None:
    """A relative Location header must be resolved against response.url before validation."""
    response = mock.MagicMock()
    response.is_redirect = True
    response.url = "http://169.254.169.254/safe-looking-path"
    response.headers = {"Location": "/latest/meta-data/"}

    with pytest.raises(ValueError):
        _ssrf_redirect_hook(response)


def test_ssrf_redirect_hook_allows_safe_redirect() -> None:
    response = mock.MagicMock()
    response.is_redirect = True
    response.url = "https://example.com/redirector"
    response.headers = {"Location": "https://example.com/final-image.png"}

    _ssrf_redirect_hook(response)  # should not raise


def test_ssrf_safe_get_rejects_blocked_url_before_request() -> None:
    with mock.patch("requests.Session.get") as mock_get:
        with pytest.raises(ValueError):
            _ssrf_safe_get("http://169.254.169.254/latest/meta-data/")
    mock_get.assert_not_called()


def test_ssrf_safe_get_registers_redirect_hook() -> None:
    fake_session = mock.MagicMock()
    fake_session.hooks = {"response": []}
    fake_session.__enter__ = mock.Mock(return_value=fake_session)
    fake_session.__exit__ = mock.Mock(return_value=False)

    with mock.patch("requests.Session", return_value=fake_session):
        _ssrf_safe_get("https://example.com/image.png")

    # The session used for the request must carry the SSRF redirect hook,
    # so that every redirect hop (not just the initial URL) gets validated.
    assert _ssrf_redirect_hook in fake_session.hooks["response"]
    fake_session.get.assert_called_once()


def test_image_node_resolve_image_blocks_ssrf_url() -> None:
    node = ImageNode(image_url="http://169.254.169.254/latest/meta-data/")
    with pytest.raises(ValueError):
        node.resolve_image()


def test_image_document_resolve_image_blocks_ssrf_url() -> None:
    # Set via the setter (post-construction) so this exercises the
    # ImageDocument.resolve_image() SSRF guard directly, rather than the
    # separate is_image_url_pil() pre-check performed in __init__.
    doc = ImageDocument()
    doc.image_url = "http://169.254.169.254/latest/meta-data/"
    with pytest.raises(ValueError):
        doc.resolve_image()
