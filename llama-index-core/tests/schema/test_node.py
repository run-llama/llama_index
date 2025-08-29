from llama_index.core.schema import MediaResource, Node, ObjectType


def test_identifiers():
    assert Node.class_name() == "Node"
    assert Node.get_type() == ObjectType.MULTIMODAL


def test_get_content():
    assert Node().get_content() == ""


def test_hash():
    node = Node()
    node.audio_resource = MediaResource(data=b"test audio", mimetype="audio/aac")
    node.image_resource = MediaResource(data=b"test image", mimetype="image/png")
    node.text_resource = MediaResource(text="some text", mimetype="text/plain")
    node.video_resource = MediaResource(data=b"some video", mimetype="video/mpeg")

    # With new JSON-based hash implementation, just verify hash properties
    assert node.hash is not None
    assert node.hash != ""
    assert isinstance(node.hash, str)
    assert len(node.hash) == 64  # SHA256 hex string length

    # Test consistency - same node setup should produce same hash
    node2 = Node()
    node2.audio_resource = MediaResource(data=b"test audio", mimetype="audio/aac")
    node2.image_resource = MediaResource(data=b"test image", mimetype="image/png")
    node2.text_resource = MediaResource(text="some text", mimetype="text/plain")
    node2.video_resource = MediaResource(data=b"some video", mimetype="video/mpeg")
    assert node.hash == node2.hash

    # Test that empty node still generates a hash
    empty_node = Node()
    assert empty_node.hash is not None
    assert empty_node.hash != ""
    assert isinstance(empty_node.hash, str)

    # Test that nodes with different resources have different hashes
    different_node = Node()
    different_node.text_resource = MediaResource(text="different text")
    assert node.hash != different_node.hash
