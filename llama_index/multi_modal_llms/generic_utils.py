import base64
from typing import List

from llama_index.schema import ImageDocument


def load_image_urls(image_urls: List[str]) -> List[ImageDocument]:
    # load remote image urls into image documents
    image_documents = []
    for i in range(len(image_urls)):
        new_image_document = ImageDocument(image_url=image_urls[i])
        image_documents.append(new_image_document)
    return image_documents


# Function to encode the image to base64 content
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
