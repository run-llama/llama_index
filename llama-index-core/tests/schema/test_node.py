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
    node.text = MediaResource(data=b"some text", mimetype="text/plain")
    node.video = MediaResource(data=b"some video", mimetype="video/mpeg")
    assert (
        node.hash == "edd73c2ed5b860a2d00d741c94f719422dc1a401c712fc551e41e084c6374e97"
    )
