from enum import Enum
from pathlib import Path
from typing import Iterable

from docling.document_converter import DocumentConverter
from docling_core.transforms.id_generator import BaseIDGenerator, DocHashIDGenerator
from docling_core.transforms.metadata_extractor import (
    BaseMetadataExtractor,
    SimpleMetadataExtractor,
)
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core import Document as LIDocument
from pydantic import Field


class DoclingReader(BasePydanticReader):
    """Docling Reader.

    Extracts PDF documents into LlamaIndex documents either using Markdown or JSON-serialized Docling native format.

    Args:
        export_type (Literal["markdown", "json"], optional): The type to export to. Defaults to "markdown".
        doc_converter (DocumentConverter, optional): The Docling converter to use. Default factory: `DocumentConverter`.
        doc_id_generator (BaseIDGenerator | None, optional): The document ID generator to use. Setting to `None` falls back to LlamaIndex's default ID generation. Defaults to `DocHashIDGenerator()`.
        metadata_extractor (BaseMetadataExtractor | None, optional): The document metadata extractor to use. Setting to `None` skips doc metadata extraction. Defaults to `SimpleMetadataExtractor()`.
    """

    class ExportType(str, Enum):
        MARKDOWN = "markdown"
        JSON = "json"

    export_type: ExportType = ExportType.MARKDOWN
    doc_converter: DocumentConverter = Field(default_factory=DocumentConverter)
    doc_id_generator: BaseIDGenerator | None = DocHashIDGenerator()
    metadata_extractor: BaseMetadataExtractor | None = SimpleMetadataExtractor()

    def lazy_load_data(
        self,
        file_path: str | Path | Iterable[str] | Iterable[Path],
        extra_info: dict | None = None,
    ) -> Iterable[LIDocument]:
        """Lazily load from given source.

        Args:
            file_path (str | Path | Iterable[str] | Iterable[Path]): Document file source as single str (URL or local file) or pathlib.Path — or iterable thereof
            extra_info (dict | None, optional): Any pre-existing metadata to include. Defaults to None.

        Returns:
            Iterable[LIDocument]: Iterable over the created LlamaIndex documents.
        """
        file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        for source in file_paths:
            dl_doc = self.doc_converter.convert_single(source).output
            text: str
            if self.export_type == self.ExportType.MARKDOWN:
                text = dl_doc.export_to_markdown()
            elif self.export_type == self.ExportType.JSON:
                text = dl_doc.model_dump_json()
            else:
                raise ValueError(f"Unexpected export type: {self.export_type}")
            origin = str(source) if isinstance(source, Path) else source
            doc_kwargs = {}
            if self.doc_id_generator:
                doc_kwargs["doc_id"] = self.doc_id_generator.generate_id(doc=dl_doc)
            if self.metadata_extractor:
                doc_kwargs[
                    "excluded_embed_metadata_keys"
                ] = self.metadata_extractor.get_excluded_embed_metadata_keys()
                doc_kwargs[
                    "excluded_llm_metadata_keys"
                ] = self.metadata_extractor.get_excluded_llm_metadata_keys()
            li_doc = LIDocument(
                text=text,
                **doc_kwargs,
            )
            li_doc.metadata = extra_info or {}
            if self.metadata_extractor:
                li_doc.metadata.update(
                    self.metadata_extractor.get_metadata(
                        doc=dl_doc,
                        origin=origin,
                    ),
                )
            yield li_doc
