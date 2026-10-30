import PIL.Image

from llama_index.core.schema import ImageDocument, MediaResource, Node, ObjectType


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


def test_media_resource_mimetype_from_uppercase_extension(tmp_path):
    resource = MediaResource(data=b"img", path=str(tmp_path / "IMG_0001.JPG"))
    assert resource.mimetype == "image/jpeg"


def test_image_document_mimetype_from_uppercase_extension(tmp_path):
    path = tmp_path / "IMG_0001.JPG"
    PIL.Image.new("RGB", (1, 1)).save(path, format="JPEG")

    doc = ImageDocument(image_path=str(path))
    assert doc.image_mimetype == "image/jpeg"
