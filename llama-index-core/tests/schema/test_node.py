from llama_index.core.schema import MediaResource, Node, ObjectType


def test_identifiers():
    assert Node.class_name() == "Node"
    assert Node.get_type() == ObjectType.MULTIMODAL


def test_get_content():
    assert Node().get_content() == ""


def test_hash():
    node = Node()
    node.audio = MediaResource(data=b"test audio", mimetype="audio/aac")
    node.image = MediaResource(data=b"test image", mimetype="image/png")
    node.text = MediaResource(text="some text", mimetype="text/plain")
    node.video = MediaResource(data=b"some video", mimetype="video/mpeg")
    assert (
        node.hash == "ee411edd3dffb27470eef165ccf4df9fabaa02e7c7c39415950d3ac4d7e35e61"
    )
