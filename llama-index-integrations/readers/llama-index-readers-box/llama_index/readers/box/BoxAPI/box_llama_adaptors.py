from box_sdk_gen import File
from llama_index.core.schema import Document


def box_file_to_llama_document_metadata(box_file: File) -> dict:
    """
    Convert a Box SDK File object to a Llama Document metadata object.

    Args:
        box_file: Box SDK File object

    Returns:
        Llama Document metadata object

    """
    # massage data
    if box_file.path_collection is not None:
        path_collection = "/".join(
            [entry.name for entry in box_file.path_collection.entries]
        )

    return {
        "box_file_id": box_file.id,
        "description": box_file.description,
        "size": box_file.size,
        "path_collection": path_collection,
        "created_at": box_file.created_at.isoformat(),
        "modified_at": box_file.modified_at.isoformat(),
        "trashed_at": box_file.trashed_at.isoformat() if box_file.trashed_at else None,
        "purged_at": box_file.purged_at.isoformat() if box_file.purged_at else None,
        "content_created_at": box_file.content_created_at.isoformat(),
        "content_modified_at": box_file.content_modified_at.isoformat(),
        "created_by": f"{box_file.created_by.id},{box_file.created_by.name},{box_file.created_by.login}",
        "modified_by": f"{box_file.modified_by.id},{box_file.modified_by.name},{box_file.modified_by.login}",
        "owned_by": f"{box_file.owned_by.id},{box_file.owned_by.name},{box_file.owned_by.login}",
        # shared_link: Optional[FileSharedLinkField] = None,
        "parent": box_file.parent.id,
        "item_status": box_file.item_status.value,
        "sequence_id": box_file.sequence_id,
        "name": box_file.name,
        "sha_1": box_file.sha_1,
        # "file_version": box_file.file_version.id,
        "etag": box_file.etag,
        "type": box_file.type.value,
    }


def box_file_to_llama_document(box_file: File) -> Document:
    """
    Convert a Box SDK File object to a Llama Document object.

    Args:
        box_file: Box SDK File object

    Returns:
        Llama Document object

    """
    document = Document()

    # this separations between extra_info and metadata seem to be pointless
    # when we update one, the other gets also updated with the same data
    document.extra_info = box_file.to_dict()
    document.metadata = box_file_to_llama_document_metadata(box_file)

    return document
