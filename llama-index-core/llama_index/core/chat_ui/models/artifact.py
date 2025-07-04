from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class ArtifactType(str, Enum):
    CODE = "code"
    DOCUMENT = "document"


class CodeArtifactData(BaseModel):
    file_name: str
    code: str
    language: str


class DocumentArtifactSource(BaseModel):
    id: str


class DocumentArtifactData(BaseModel):
    title: str
    content: str
    type: Literal["markdown", "html"]
    sources: Optional[List[DocumentArtifactSource]] = None


class Artifact(BaseModel):
    created_at: Optional[int] = None
    type: ArtifactType
    data: Union[CodeArtifactData, DocumentArtifactData]
