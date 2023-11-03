from llama_index.readers import SimpleDirectoryReader
from llama_index.schema import ImageDocument


def load_image_urls(image_urls: list[str]) -> list[ImageDocument]:
    image_documents = []
    for i in range(len(image_urls)):
        new_image_document = ImageDocument()
        new_image_document.metadata["image_url"] = image_urls[i]
        image_documents.append(new_image_document)
    return image_documents


def load_local_image_files_from_folder(image_dir_path: str) -> list[ImageDocument]:
    return SimpleDirectoryReader(image_dir_path).load_data()
