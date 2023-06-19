from pydantic import BaseModel, Field
from typing import List
import enum


class NodeType(str, enum.Enum):
    """Enumeration representing the types of nodes in a filesystem."""

    FILE = "file"
    FOLDER = "folder"


class Node(BaseModel):
    """
    Class representing a single node in a filesystem. Can be either a file or a folder.
    Note that a file cannot have children, but a folder can.

    Args:
        name (str): The name of the node.
        children (List[Node]): The list of child nodes (if any).
        node_type (NodeType): The type of the node, either a file or a folder.

    Methods:
        print_paths: Prints the path of the node and its children.
    """

    name: str = Field(..., description="Name of the folder")
    children: List["Node"] = Field(
        default_factory=list,
        description="List of children nodes, only applicable for folders, files cannot have children",
    )
    node_type: NodeType = Field(
        default=NodeType.FILE,
        description="Either a file or folder, use the name to determine which it could be",
    )


class DirectoryTree(BaseModel):
    """
    Container class representing a directory tree.

    Args:
        root (Node): The root node of the tree.

    Methods:
        print_paths: Prints the paths of the root node and its children.
    """

    root: Node = Field(..., description="Root folder of the directory tree")
