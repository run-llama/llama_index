"""Docugami reader."""

import io
import os
import re
from typing import Any, Dict, List, Mapping, Optional

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

TD_NAME = "{http://www.w3.org/1999/xhtml}td"
TABLE_NAME = "{http://www.w3.org/1999/xhtml}table"

XPATH_KEY = "xpath"
DOCUMENT_ID_KEY = "id"
DOCUMENT_NAME_KEY = "name"
STRUCTURE_KEY = "structure"
TAG_KEY = "tag"
PROJECTS_KEY = "projects"

DEFAULT_API_ENDPOINT = "https://api.docugami.com/v1preview1"


class DocugamiReader(BaseReader):
    """Docugami reader.

    Reads Documents as nodes in a Document XML Knowledge Graph, from Docugami.

    """

    api: str = DEFAULT_API_ENDPOINT
    access_token: Optional[str] = os.environ.get("DOCUGAMI_API_KEY")
    min_chunk_size: int = 32  # appended to next chunk to avoid over-chunking

    def _parse_dgml(
        self, document: Mapping, content: bytes, doc_metadata: Optional[Mapping] = None
    ) -> List[Document]:
        """Parse a single DGML document into a list of Documents."""
        try:
            from lxml import etree
        except ImportError:
            raise ValueError(
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
            structure = (
                "table"
                if node.tag == TABLE_NAME
                else node.attrib["structure"]
                if "structure" in node.attrib
                else None
            )
            return structure

        def _is_structural(node: Any) -> bool:
            """Check if a node is structural."""
            return _structure_value(node) is not None

        def _is_heading(node: Any) -> bool:
            """Check if a node is a heading."""
            structure = _structure_value(node)
            return structure is not None and structure.lower().startswith("h")

        def _get_text(node: Any) -> str:
            """Get the text of a node."""
            return " ".join(node.itertext()).strip()

        def _has_structural_descendant(node: Any) -> bool:
            """Check if a node has a structural descendant."""
            for child in node:
                if _is_structural(child) or _has_structural_descendant(child):
                    return True
            return False

        def _leaf_structural_nodes(node: Any) -> List:
            """Get the leaf structural nodes of a node."""
            if _is_structural(node) and not _has_structural_descendant(node):
                return [node]
            else:
                leaf_nodes = []
                for child in node:
                    leaf_nodes.extend(_leaf_structural_nodes(child))
                return leaf_nodes

        def _create_doc(node: Any, text: str) -> Document:
            """Create a Document from a node and text."""
            metadata = {
                XPATH_KEY: _xpath_for_chunk(node),
                DOCUMENT_ID_KEY: document["id"],
                DOCUMENT_NAME_KEY: document["name"],
                STRUCTURE_KEY: node.attrib.get("structure", ""),
                TAG_KEY: re.sub(r"\{.*\}", "", node.tag),
            }

            if doc_metadata:
                metadata.update(doc_metadata)

            return Document(
                text=text,
                metadata=metadata,
                excluded_llm_metadata_keys=[XPATH_KEY, DOCUMENT_ID_KEY, STRUCTURE_KEY],
            )

        # parse the tree and return chunks
        tree = etree.parse(io.BytesIO(content))
        root = tree.getroot()

        chunks: List[Document] = []
        prev_small_chunk_text = None
        for node in _leaf_structural_nodes(root):
            text = _get_text(node)
            if prev_small_chunk_text:
                text = prev_small_chunk_text + " " + text
                prev_small_chunk_text = None

            if _is_heading(node) or len(text) < self.min_chunk_size:
                # Save headings or other small chunks to be appended to the next chunk
                prev_small_chunk_text = text
            else:
                chunks.append(_create_doc(node, text))

        if prev_small_chunk_text and len(chunks) > 0:
            # small chunk at the end left over, just append to last chunk
            if not chunks[-1].text:
                chunks[-1].text = prev_small_chunk_text
            else:
                chunks[-1].text += " " + prev_small_chunk_text

        return chunks

    def _document_details_for_docset_id(self, docset_id: str) -> List[Dict]:
        """Gets all document details for the given docset ID"""
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
        """Gets all project details for the given docset ID"""
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
        """Gets project metadata for all files"""
        project_id = project.get("id")

        url = f"{self.api}/projects/{project_id}/artifacts/latest"
        all_artifacts = []

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
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        per_file_metadata = {}
        for artifact in all_artifacts:
            artifact_name = artifact.get("name")
            artifact_url = artifact.get("url")
            artifact_doc = artifact.get("document")

            if artifact_name == "report-values.xml" and artifact_url and artifact_doc:
                doc_id = artifact_doc["id"]
                metadata: Dict = {}

                # the evaluated XML for each document is named after the project
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
                        raise ValueError(
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
                        metadata[heading] = value
                    per_file_metadata[doc_id] = metadata
                else:
                    raise Exception(
                        f"Failed to download {artifact_url}/content "
                        + "(status: {response.status_code})"
                    )

        return per_file_metadata

    def _load_chunks_for_document(
        self, docset_id: str, document: Dict, doc_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Load chunks for a document."""
        document_id = document["id"]
        url = f"{self.api}/docsets/{docset_id}/documents/{document_id}/dgml"

        response = requests.request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )

        if response.ok:
            return self._parse_dgml(document, response.content, doc_metadata)
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
        """Load data the given docset_id in Docugami

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
                d for d in _document_details if d["id"] in document_ids
            ]

        _project_details = self._project_details_for_docset_id(docset_id)
        combined_project_metadata = {}
        if _project_details:
            # if there are any projects for this docset, load project metadata
            for project in _project_details:
                metadata = self._metadata_for_project(project)
                combined_project_metadata.update(metadata)

        for doc in _document_details:
            doc_metadata = combined_project_metadata.get(doc["id"])
            chunks += self._load_chunks_for_document(docset_id, doc, doc_metadata)

        return chunks


if __name__ == "__main__":
    reader = DocugamiReader()
    print(
        reader.load_data(
            docset_id="ecxqpipcoe2p", document_ids=["43rj0ds7s0ur", "bpc1vibyeke2"]
        )
    )
