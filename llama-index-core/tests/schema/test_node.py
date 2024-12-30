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
    assert (
        node.hash == "6f08712269634de7e53e62a3aaee59d60e9a32a43bc05284a21244f960f0cda4"
    )
