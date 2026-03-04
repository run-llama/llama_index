"""Docugami reader."""

import hashlib
import io
import logging
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from dgml_utils.models import Chunk
from dgml_utils.segmentation import get_chunks

TABLE_NAME = "{http://www.w3.org/1999/xhtml}table"

XPATH_KEY = "xpath"
ID_KEY = "id"
DOCUMENT_SOURCE_KEY = "source"
DOCUMENT_NAME_KEY = "name"
STRUCTURE_KEY = "structure"
TAG_KEY = "tag"
PROJECTS_KEY = "projects"

DEFAULT_API_ENDPOINT = "https://api.docugami.com/v1preview1"

logger = logging.getLogger(__name__)


class DocugamiReader(BaseReader):
    """
    Docugami reader.

    Reads Documents as nodes in a Document XML Knowledge Graph, from Docugami.

    """

    api: str = DEFAULT_API_ENDPOINT
    """The Docugami API endpoint to use."""

    access_token: Optional[str] = os.environ.get("DOCUGAMI_API_KEY")
    """The Docugami API access token to use."""

    max_text_length = 4096
    """Max length of chunk text returned."""

    min_text_length: int = 32
    """Threshold under which chunks are appended to next to avoid over-chunking."""

    max_metadata_length = 512
    """Max length of metadata text returned."""

    include_xml_tags: bool = False
    """Set to true for XML tags in chunk output text."""

    parent_hierarchy_levels: int = 0
    """Set appropriately to get parent chunks using the chunk hierarchy."""

    parent_id_key: str = "doc_id"
    """Metadata key for parent doc ID."""

    sub_chunk_tables: bool = False
    """Set to True to return sub-chunks within tables."""

    whitespace_normalize_text: bool = True
    """Set to False if you want to full whitespace formatting in the original
    XML doc, including indentation."""

    docset_id: Optional[str]
    """The Docugami API docset ID to use."""

    document_ids: Optional[Sequence[str]]
    """The Docugami API document IDs to use."""

    file_paths: Optional[Sequence[Union[Path, str]]]
    """The local file paths to use."""

    include_project_metadata_in_doc_metadata: bool = True
    """Set to True if you want to include the project metadata in the doc metadata."""

    def __init__(
        self,
        api: str = DEFAULT_API_ENDPOINT,
        access_token: Optional[str] = os.environ.get("DOCUGAMI_API_KEY"),
        max_text_length=4096,
        min_text_length: int = 32,
        max_metadata_length=512,
        include_xml_tags: bool = False,
        parent_hierarchy_levels: int = 0,
        parent_id_key: str = "doc_id",
        sub_chunk_tables: bool = False,
        whitespace_normalize_text: bool = True,
        docset_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
        file_paths: Optional[Sequence[Union[Path, str]]] = None,
        include_project_metadata_in_doc_metadata: bool = True,
    ):
        self.api = api
        self.access_token = access_token
        self.max_text_length = max_text_length
        self.min_text_length = min_text_length
        self.max_metadata_length = max_metadata_length
        self.include_xml_tags = include_xml_tags
        self.parent_hierarchy_levels = parent_hierarchy_levels
        self.parent_id_key = parent_id_key
        self.sub_chunk_tables = sub_chunk_tables
        self.whitespace_normalize_text = whitespace_normalize_text
        self.docset_id = docset_id
        self.document_ids = document_ids
        self.file_paths = file_paths
        self.include_project_metadata_in_doc_metadata = (
            include_project_metadata_in_doc_metadata
        )

    def _parse_dgml(
        self,
        content: bytes,
        document_name: Optional[str] = None,
        additional_doc_metadata: Optional[Mapping] = None,
    ) -> List[Document]:
        """Parse a single DGML document into a list of Documents."""
        try:
            from lxml import etree
        except ImportError:
            raise ImportError(
                "Could not import lxml python package. "
                "Please install it with `pip install lxml`."
            )

        # helpers
        def _xpath_qname_for_chunk(chunk: Any) -> str:
            """Get the xpath qname for a chunk."""
            qname = f"{chunk.prefix}:{chunk.tag.split('}')[-1]}"

            parent = chunk.getparent()
            if parent is not None:
                doppelgangers = [x for x in parent if x.tag == chunk.tag]
                if len(doppelgangers) > 1:
                    idx_of_self = doppelgangers.index(chunk)
                    qname = f"{qname}[{idx_of_self + 1}]"

            return qname

        def _xpath_for_chunk(chunk: Any) -> str:
            """Get the xpath for a chunk."""
            ancestor_chain = chunk.xpath("ancestor-or-self::*")
            return "/" + "/".join(_xpath_qname_for_chunk(x) for x in ancestor_chain)

        def _structure_value(node: Any) -> Optional[str]:
            """Get the structure value for a node."""
            return (
                "table"
                if node.tag == TABLE_NAME
                else node.attrib["structure"]
                if "structure" in node.attrib
                else None
            )

        def _build_framework_chunk(dg_chunk: Chunk) -> Document:
            # Adding dg_chunk.text + dg_chunk.xpath should prevent hash collision between two chunks that have the same text but a different xpath
            text = dg_chunk.xpath + "\n" + dg_chunk.text
            _hashed_id = hashlib.md5(text.encode()).hexdigest()
            metadata = {
                XPATH_KEY: dg_chunk.xpath,
                ID_KEY: _hashed_id,
                DOCUMENT_NAME_KEY: document_name,
                STRUCTURE_KEY: dg_chunk.structure,
                TAG_KEY: dg_chunk.tag,
            }

            text = dg_chunk.text
            if additional_doc_metadata:
                if self.include_project_metadata_in_doc_metadata:
                    metadata.update(additional_doc_metadata)

            return Document(
                text=text[: self.max_text_length],
                metadata=metadata,
                excluded_llm_metadata_keys=[XPATH_KEY, ID_KEY, STRUCTURE_KEY],
            )

        # Parse the tree and return chunks
        tree = etree.parse(io.BytesIO(content))
        root = tree.getroot()

        dg_chunks = get_chunks(
            root,
            min_text_length=self.min_text_length,
            max_text_length=self.max_text_length,
            whitespace_normalize_text=self.whitespace_normalize_text,
            sub_chunk_tables=self.sub_chunk_tables,
            include_xml_tags=self.include_xml_tags,
            parent_hierarchy_levels=self.parent_hierarchy_levels,
        )

        framework_chunks: Dict[str, Document] = {}
        for dg_chunk in dg_chunks:
            framework_chunk = _build_framework_chunk(dg_chunk)
            chunk_id = framework_chunk.metadata.get(ID_KEY)
            if chunk_id:
                framework_chunks[chunk_id] = framework_chunk
                if dg_chunk.parent:
                    framework_parent_chunk = _build_framework_chunk(dg_chunk.parent)
                    parent_id = framework_parent_chunk.metadata.get(ID_KEY)
                    if parent_id and framework_parent_chunk.text:
                        framework_chunk.metadata[self.parent_id_key] = parent_id
                        framework_chunks[parent_id] = framework_parent_chunk

        return list(framework_chunks.values())

    def _document_details_for_docset_id(self, docset_id: str) -> List[Dict]:
        """Gets all document details for the given docset ID."""
        url = f"{self.api}/docsets/{docset_id}/documents"
        all_documents = []

        while url:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.access_token}"},
            )
            if response.ok:
                data = response.json()
                all_documents.extend(data["documents"])
                url = data.get("next", None)
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        return all_documents

    def _project_details_for_docset_id(self, docset_id: str) -> List[Dict]:
        """Gets all project details for the given docset ID."""
        url = f"{self.api}/projects?docset.id={docset_id}"
        all_projects = []

        while url:
            response = requests.request(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                data={},
            )
            if response.ok:
                data = response.json()
                all_projects.extend(data["projects"])
                url = data.get("next", None)
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        return all_projects

    def _metadata_for_project(self, project: Dict) -> Dict:
        """Gets project metadata for all files."""
        project_id = project.get(ID_KEY)

        url = f"{self.api}/projects/{project_id}/artifacts/latest"
        all_artifacts = []

        per_file_metadata: Dict = {}
        while url:
            response = requests.request(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                data={},
            )
            if response.ok:
                data = response.json()
                all_artifacts.extend(data["artifacts"])
                url = data.get("next", None)
            elif response.status_code == 404:
                # Not found is ok, just means no published projects
                return per_file_metadata
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        for artifact in all_artifacts:
            artifact_name = artifact.get("name")
            artifact_url = artifact.get("url")
            artifact_doc = artifact.get("document")

            if artifact_name == "report-values.xml" and artifact_url and artifact_doc:
                doc_id = artifact_doc[ID_KEY]
                metadata: Dict = {}

                # The evaluated XML for each document is named after the project
                response = requests.request(
                    "GET",
                    f"{artifact_url}/content",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    data={},
                )

                if response.ok:
                    try:
                        from lxml import etree
                    except ImportError:
                        raise ImportError(
                            "Could not import lxml python package. "
                            "Please install it with `pip install lxml`."
                        )
                    artifact_tree = etree.parse(io.BytesIO(response.content))
                    artifact_root = artifact_tree.getroot()
                    ns = artifact_root.nsmap
                    entries = artifact_root.xpath("//pr:Entry", namespaces=ns)
                    for entry in entries:
                        heading = entry.xpath("./pr:Heading", namespaces=ns)[0].text
                        value = " ".join(
                            entry.xpath("./pr:Value", namespaces=ns)[0].itertext()
                        ).strip()
                        metadata[heading] = value[: self.max_metadata_length]
                    per_file_metadata[doc_id] = metadata
                else:
                    raise Exception(
                        f"Failed to download {artifact_url}/content "
                        + "(status: {response.status_code})"
                    )

        return per_file_metadata

    def _load_chunks_for_document(
        self,
        document_id: str,
        docset_id: str,
        document_name: Optional[str] = None,
        additional_metadata: Optional[Mapping] = None,
    ) -> List[Document]:
        """Load chunks for a document."""
        url = f"{self.api}/docsets/{docset_id}/documents/{document_id}/dgml"

        response = requests.request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )

        if response.ok:
            return self._parse_dgml(
                content=response.content,
                document_name=document_name,
                additional_doc_metadata=additional_metadata,
            )
        else:
            raise Exception(
                f"Failed to download {url} (status: {response.status_code})"
            )

    def load_data(
        self,
        docset_id: str,
        document_ids: Optional[List[str]] = None,
        access_token: Optional[str] = None,
    ) -> List[Document]:
        """
        Load data the given docset_id in Docugami.

        Args:
            docset_id (str): Document set ID to load data for.
            document_ids (Optional[List[str]]): Optional list of document ids to load data for.
                                    If not specified, all documents from docset_id are loaded.

        """
        chunks: List[Document] = []

        if access_token:
            self.access_token = access_token

        if not self.access_token:
            raise Exception(
                "Please specify access token as argument or set the DOCUGAMI_API_KEY"
                " env var."
            )

        _document_details = self._document_details_for_docset_id(docset_id)
        if document_ids:
            _document_details = [
                d for d in _document_details if d[ID_KEY] in document_ids
            ]

        _project_details = self._project_details_for_docset_id(docset_id)
        combined_project_metadata: Dict[str, Dict] = {}
        if _project_details and self.include_project_metadata_in_doc_metadata:
            # If there are any projects for this docset and the caller requested
            # project metadata, load it.
            for project in _project_details:
                metadata = self._metadata_for_project(project)
                for file_id in metadata:
                    if file_id not in combined_project_metadata:
                        combined_project_metadata[file_id] = metadata[file_id]
                    else:
                        combined_project_metadata[file_id].update(metadata[file_id])

        for doc in _document_details:
            doc_id = doc[ID_KEY]
            doc_name = doc.get(DOCUMENT_NAME_KEY)
            doc_metadata = combined_project_metadata.get(doc_id)
            chunks += self._load_chunks_for_document(
                document_id=doc_id,
                docset_id=docset_id,
                document_name=doc_name,
                additional_metadata=doc_metadata,
            )

        return chunks
