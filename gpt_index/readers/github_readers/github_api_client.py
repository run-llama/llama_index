from dataclasses import dataclass
from typing import List, Optional

from dataclasses_json import DataClassJsonMixin


@dataclass
class GitTreeResponseModel(DataClassJsonMixin):
    """
    Dataclass for the response from the Github API's getTree endpoint.

    Attributes:
        - sha (str): SHA1 checksum ID of the tree.
        - url (str): URL for the tree.
        - tree (List[GitTreeObject]): List of objects in the tree.
        - truncated (bool): Whether the tree is truncated.

    Examples:
        >>> tree = client.get_tree("owner", "repo", "branch")
        >>> tree.sha
    """

    @dataclass
    class GitTreeObject(DataClassJsonMixin):
        """
        Dataclass for the objects in the tree.

        Attributes:
            - path (str): Path to the object.
            - mode (str): Mode of the object.
            - type (str): Type of the object.
            - sha (str): SHA1 checksum ID of the object.
            - url (str): URL for the object.
            - size (Optional[int]): Size of the object (only for blobs).
        """

        path: str
        mode: str
        type: str
        sha: str
        url: str
        size: Optional[int] = None

    sha: str
    url: str
    tree: List[GitTreeObject]
    truncated: bool


@dataclass
class GitBlobResponseModel(DataClassJsonMixin):
    """
    Dataclass for the response from the Github API's getBlob endpoint.

    Attributes:
        - content (str): Content of the blob.
        - encoding (str): Encoding of the blob.
        - url (str): URL for the blob.
        - sha (str): SHA1 checksum ID of the blob.
        - size (int): Size of the blob.
        - node_id (str): Node ID of the blob.
    """

    content: str
    encoding: str
    url: str
    sha: str
    size: int
    node_id: str
