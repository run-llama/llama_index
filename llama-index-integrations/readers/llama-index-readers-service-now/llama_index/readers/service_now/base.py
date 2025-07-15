from enum import Enum
import logging
import os
from typing import Callable, Dict, List, Optional
import uuid
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pysnc import GlideRecord
import requests

from .event import AttachmentEvent, Event, EventName, PageEvent

internal_logger = logging.getLogger(__name__)

DEFAULT_WORKFLOW_STATE = "Published"


class FileType(Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    TEXT = "text"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "md"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    PDF = "pdf"
    UNKNOWN = "unknown"


class Observer:
    def __init__(self):
        self._listeners: Dict[EventName, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_name: EventName, callback: Callable[[Event], None]):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)

    def subscribe_all(self, callback: Callable[[Event], None]):
        for event_name in EventName:
            if event_name not in self._listeners:
                self._listeners[event_name] = []
            self._listeners[event_name].append(callback)

    def unsubscribe(self, event_name: EventName, callback: Callable[[Event], None]):
        if event_name in self._listeners:
            self._listeners[event_name].remove(callback)
            if not self._listeners[event_name]:
                del self._listeners[event_name]

    def notify(self, event: Event):
        if event.name in self._listeners:
            for callback in self._listeners[event.name]:
                callback(event)


class CustomParserManager:
    def __init__(
        self,
        custom_parsers: Dict[FileType, BaseReader],
        custom_folder: str,
        logger=None,
    ):
        if not custom_parsers:
            raise ValueError("custom_parsers is required and cannot be empty")

        if not isinstance(custom_parsers, dict):
            raise TypeError("custom_parsers must be a dictionary")

        # Validate custom_parsers dictionary
        for file_type, parser in custom_parsers.items():
            if not isinstance(file_type, FileType):
                raise TypeError(
                    f"custom_parsers keys must be FileType enum values, got {type(file_type)}"
                )

            if not isinstance(parser, BaseReader):
                raise TypeError(
                    f"custom_parsers values must be BaseReader instances, got {type(parser)} for {file_type}"
                )

        self.custom_parsers = custom_parsers
        self.custom_folder = custom_folder
        self.logger = logger or internal_logger

        # Ensure custom_folder exists and is writable
        try:
            os.makedirs(self.custom_folder, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(self.custom_folder, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, PermissionError) as e:
            raise ValueError(
                f"Custom folder '{self.custom_folder}' is not accessible or writable: {e}"
            )

    @staticmethod
    def validate_recommended_parsers(
        custom_parsers: Dict[FileType, BaseReader], logger=None
    ) -> List[str]:
        """
        Validate that custom parsers are provided for recommended file types.

        Args:
            custom_parsers: Dictionary of file type to parser mappings
            logger: Optional logger for warnings

        Returns:
            List of missing recommended file types

        """
        recommended_types = [
            FileType.PDF,
            FileType.HTML,
            FileType.DOCUMENT,
            FileType.TEXT,
            FileType.SPREADSHEET,
            FileType.PRESENTATION,
        ]

        missing_types = []
        for file_type in recommended_types:
            if file_type not in custom_parsers:
                missing_types.append(file_type.value)

        if missing_types and logger:
            logger.warning(
                f"Recommended custom parsers missing for file types: {', '.join(missing_types)}"
            )

        return missing_types

    def __remove_custom_file(self, file_path: str):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            self.logger.error(f"Error removing file {file_path}: {e}")

    def process_with_custom_parser(
        self, file_type: FileType, file_content: bytes, extension: str
    ) -> str:
        """
        Process file content with a custom parser (required).

        Args:
            file_type: The type of file to process
            file_content: The binary file content to process
            extension: The file extension

        Returns:
            Processed markdown text

        Raises:
            ValueError: If no custom parser found for file type or content is empty

        """
        if file_type not in self.custom_parsers:
            raise ValueError(f"No custom parser found for file type: {file_type}")

        if not file_content:
            raise ValueError("File content cannot be empty")

        file_name = f"{uuid.uuid4().hex}.{extension}"
        custom_file_path = os.path.join(self.custom_folder, file_name)

        try:
            with open(custom_file_path, "wb") as f:
                f.write(file_content)

            markdown_text = ""
            try:
                documents = self.custom_parsers[file_type].load_data(
                    file_path=custom_file_path
                )
                if not documents:
                    raise ValueError(
                        f"Custom parser for {file_type} returned no documents"
                    )

                markdown_text = "\n".join(doc.text for doc in documents)

                if not markdown_text.strip():
                    raise ValueError(
                        f"Custom parser for {file_type} returned empty content"
                    )

            except Exception as e:
                raise ValueError(
                    f"Error processing file with custom parser for {file_type}: {e}"
                )
            finally:
                self.__remove_custom_file(custom_file_path)

            return markdown_text

        except (OSError, PermissionError) as e:
            raise ValueError(f"Error creating temporary file for parsing: {e}")

    def process_text_with_custom_parser(
        self, file_type: FileType, text_content: str, extension: str
    ) -> str:
        """
        Process text content with a custom parser (required).

        Args:
            file_type: The type of file to process
            text_content: The text content to process
            extension: The file extension

        Returns:
            Processed markdown text

        Raises:
            ValueError: If no custom parser found for file type or content is empty

        """
        if file_type not in self.custom_parsers:
            raise ValueError(f"No custom parser found for file type: {file_type}")

        if not text_content:
            raise ValueError("Text content cannot be empty")

        # Create a temporary file-like object
        file_name = f"{uuid.uuid4().hex}.{extension}"
        custom_file_path = os.path.join(self.custom_folder, file_name)

        try:
            with open(custom_file_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            markdown_text = ""
            try:
                documents = self.custom_parsers[file_type].load_data(
                    file_path=custom_file_path
                )
                if not documents:
                    raise ValueError(
                        f"Custom parser for {file_type} returned no documents"
                    )

                markdown_text = "\n".join(doc.text for doc in documents)

                if not markdown_text.strip():
                    raise ValueError(
                        f"Custom parser for {file_type} returned empty content"
                    )

            except Exception as e:
                raise ValueError(
                    f"Error processing text with custom parser for {file_type}: {e}"
                )
            finally:
                self.__remove_custom_file(custom_file_path)

            return markdown_text

        except (OSError, PermissionError) as e:
            raise ValueError(f"Error creating temporary file for parsing: {e}")


class SnowKBReader(BaseReader):
    """
    ServiceNow Knowledge Base reader using PySNC with username/password or password grant flow.

    This reader requires custom parsers for processing different file types. Custom parsers
    must be provided for each file type you expect to encounter in your ServiceNow knowledge base.

    Recommended file types to provide parsers for:
    - FileType.PDF: For PDF documents
    - FileType.HTML: For HTML content (required for article body processing)
    - FileType.DOCUMENT: For Word documents (.docx)
    - FileType.TEXT: For plain text files
    - FileType.SPREADSHEET: For Excel files (.xlsx)
    - FileType.PRESENTATION: For PowerPoint files (.pptx)

    Args:
        instance: ServiceNow instance name (without .service-now.com)
        custom_parsers: Dictionary mapping FileType enum values to BaseReader instances.
                       This is REQUIRED and cannot be empty. Each parser must implement
                       the load_data method.
        username: ServiceNow username for authentication (required)
        password: ServiceNow password for authentication (required)
        client_id: OAuth client ID for ServiceNow (optional, but if provided, client_secret is required)
        client_secret: OAuth client secret for ServiceNow (optional, but if provided, client_id is required)
        process_attachment_callback: Optional callback to filter attachments
        process_document_callback: Optional callback to filter documents
        custom_folder: Folder for temporary files during parsing
        fail_on_error: Whether to fail on parsing errors or continue
        kb_table: ServiceNow table name for knowledge base articles
        logger: Optional logger instance

    Authentication:
        - Basic auth: Provide username and password only
        - OAuth flow: Provide username, password, client_id, and client_secret

    Raises:
        ValueError: If required parameters are missing or invalid
        TypeError: If custom_parsers types are incorrect

    """

    def __init__(
        self,
        instance: str,
        custom_parsers: Dict[FileType, BaseReader],
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        process_attachment_callback: Optional[
            Callable[[str, int], tuple[bool, str]]
        ] = None,
        process_document_callback: Optional[Callable[[str], bool]] = None,
        custom_folder: Optional[str] = None,
        fail_on_error: bool = True,
        kb_table: str = "kb_knowledge",
        logger=None,
    ):
        # Validate required parameters
        if not instance:
            raise ValueError("instance parameter is required")

        if custom_parsers is None:
            raise ValueError("custom_parsers parameter is required and cannot be None")

        if not custom_parsers:
            raise ValueError("custom_parsers parameter is required and cannot be empty")

        if not isinstance(custom_parsers, dict):
            raise TypeError("custom_parsers must be a dictionary")

        # Validate custom_parsers dictionary - ensure it has at least one parser
        if len(custom_parsers) == 0:
            raise ValueError("custom_parsers must contain at least one parser")

        # Validate each custom parser
        for file_type, parser in custom_parsers.items():
            if not isinstance(file_type, FileType):
                raise TypeError(
                    f"custom_parsers keys must be FileType enum values, got {type(file_type)}"
                )

            if not isinstance(parser, BaseReader):
                raise TypeError(
                    f"custom_parsers values must be BaseReader instances, got {type(parser)} for {file_type}"
                )

            # Validate that parser has required load_data method
            if not hasattr(parser, "load_data") or not callable(parser.load_data):
                raise TypeError(
                    f"custom_parsers[{file_type}] must have a callable 'load_data' method"
                )

        # Validate authentication parameters
        # Username and password are always required
        if not username:
            raise ValueError("username parameter is required")
        if not password:
            raise ValueError("password parameter is required")

        # If client_id is provided, client_secret must also be provided (for OAuth flow)
        if client_id is not None and client_secret is None:
            raise ValueError("client_secret is required when client_id is provided")
        if client_secret is not None and client_id is None:
            raise ValueError("client_id is required when client_secret is provided")

        self.instance = instance
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret

        self.custom_parsers = custom_parsers
        self.custom_folder = custom_folder or os.path.join(
            os.getcwd(), "custom_parsers"
        )

        # Validate recommended parsers and warn if missing
        self.logger = logger or internal_logger
        CustomParserManager.validate_recommended_parsers(custom_parsers, self.logger)

        # Ensure custom_folder exists and is writable
        try:
            os.makedirs(self.custom_folder, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(self.custom_folder, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, PermissionError) as e:
            raise ValueError(
                f"Custom folder '{self.custom_folder}' is not accessible or writable: {e}"
            )

        self.process_attachment_callback = process_attachment_callback
        self.process_document_callback = process_document_callback
        self.fail_on_error = fail_on_error

        self.kb_table = kb_table
        self.pysnc_client = None

        self.initialize_client()

        self.observer = Observer()
        self.custom_parser_manager = CustomParserManager(
            custom_parsers, self.custom_folder, self.logger
        )

    def _format_attachment_header(self, attachment: dict) -> str:
        """Formats the attachment title as a markdown header."""
        return f"# {attachment['file_name']}\n"

    def initialize_client(self):
        """Initialize a new ServiceNowClient instance with fresh credentials."""
        try:
            from pysnc import ServiceNowClient
            from pysnc.auth import ServiceNowPasswordGrantFlow
        except ImportError:
            self.logger.error(
                "`pysnc` package not found, please run `pip install pysnc`"
            )
            raise ImportError(
                "`pysnc` package not found, please run `pip install pysnc`"
            )
        try:
            self.logger.info("Initializing ServiceNow client")
            instance = self.instance
            user = self.username
            password = self.password

            # Use OAuth flow if client_id and client_secret are provided, otherwise use basic auth
            if self.client_id and self.client_secret:
                client_id = self.client_id
                client_secret = self.client_secret
                self.pysnc_client = ServiceNowClient(
                    instance,
                    ServiceNowPasswordGrantFlow(
                        user, password, client_id, client_secret
                    ),
                )
            else:
                # Basic authentication with username and password
                self.pysnc_client = ServiceNowClient(instance, (user, password))
        except Exception as e:
            self.logger.error(f"Error initializing ServiceNow client: {e}")
            raise ValueError(f"Error initializing ServiceNow client: {e}")

    def get_documents(
        self,
        article_sys_id: Optional[str] = None,
        numbers: Optional[List[str]] = None,
        status: str = DEFAULT_WORKFLOW_STATE,
        metadata_only: bool = False,
    ) -> List[Document]:
        """
        Fetch a KB article by sys_id or number and return a list of Documents. Also downloads attachments.
        """
        gr = self.pysnc_client.GlideRecord(self.kb_table)
        if article_sys_id:
            gr.add_query("sys_id", article_sys_id)
        elif numbers:
            gr.add_query("number", "IN", ",".join(numbers))
        else:
            raise ValueError("Must provide article_sys_id or number")
        gr.add_query("latest", "true")  # Only fetch latest version
        gr.add_query(
            "workflow_state", status or DEFAULT_WORKFLOW_STATE
        )  # Include only published articles
        gr.query()
        if not gr.has_next():
            self.logger.error(
                f"No KB article found for sys_id {article_sys_id} or numbers {numbers}"
            )
            raise ValueError(
                f"No KB article found for sys_id {article_sys_id} or numbers {numbers}"
            )
        docs = []

        total_pages = gr.get_row_count()
        self.logger.info(
            f"Found {total_pages} KB articles matching criteria: sys_id={article_sys_id}, numbers={numbers}, status={status}"
        )
        self.observer.notify(
            PageEvent(
                name=EventName.TOTAL_PAGES_TO_PROCESS,
                page_id="",
                document={},
                metadata={"total_pages": total_pages},
            )
        )

        while gr.next():
            try:
                self.observer.notify(
                    PageEvent(
                        name=EventName.PAGE_DATA_FETCH_STARTED,
                        page_id=gr.number.get_value(),
                        document={},
                    )
                )

                # Initialize complete_text for both metadata_only and full content cases
                complete_text = ""
                if not metadata_only:
                    txt_lm = (
                        gr.article_body
                        if hasattr(gr, "article_body") and gr.article_body
                        else gr.text.get_value()
                    )
                    attachments = self.handle_attachments(
                        gr.sys_id.get_value(), kb_number=gr.number.get_value()
                    )

                    try:
                        article_markdown = (
                            self.custom_parser_manager.process_text_with_custom_parser(
                                FileType.HTML, txt_lm, "html"
                            )
                        )
                    except ValueError as e:
                        self.logger.error(
                            f"Error processing article HTML with custom parser: {e}"
                        )
                        if self.fail_on_error:
                            raise
                        article_markdown = txt_lm  # Fallback to original text

                    complete_text = (
                        article_markdown
                        + "\n\n"
                        + "\n".join(
                            self._format_attachment_header(attach)
                            + attach["markdown_text"]
                            for attach in attachments
                            if "markdown_text" in attach
                        )
                    )

                display_number = (
                    gr.get_display_value("display_number")
                    if hasattr(gr, "display_number")
                    else None
                )
                sys_updated_on = (
                    gr.get_value("sys_updated_on")
                    if hasattr(gr, "sys_updated_on")
                    else None
                )
                kb_number = gr.get_value("number") if hasattr(gr, "number") else None
                kb_status = (
                    gr.workflow_state.get_display_value()
                    if hasattr(gr, "workflow_state")
                    else "Unknown"
                )

                doc = Document(
                    text=complete_text,
                    extra_info={
                        "title": gr.short_description.get_display_value()
                        if hasattr(gr, "short_description")
                        else "No Title",
                        "page_id": kb_number,
                        "status": kb_status,
                        "version": display_number,
                        "sys_updated_on": sys_updated_on,
                        "kb_number": kb_number,
                    },
                )
                metadata = {
                    "version": display_number,
                    "sys_updated_on": sys_updated_on,
                    "kb_number": kb_number,
                }
                self.observer.notify(
                    PageEvent(
                        name=EventName.PAGE_DATA_FETCH_COMPLETED,
                        page_id=kb_number,
                        document=doc,
                        metadata=metadata,
                    )
                )
                docs.append(doc)
            except Exception as e:
                self.logger.error(
                    f"Error processing KB article {gr.number.get_value()}: {e}"
                )
                self.observer.notify(
                    PageEvent(
                        name=EventName.PAGE_FAILED,
                        page_id=gr.number.get_value(),
                        document={},
                        error=str(e),
                    )
                )
                if self.fail_on_error:
                    raise
        return docs

    def get_attachment_event(
        self, event_name: EventName, page_id: str, gr_attach: GlideRecord, error=""
    ):
        return AttachmentEvent(
            page_id=page_id,
            name=event_name,
            attachment_id=f"{gr_attach.get_value('sys_id')}",
            attachment_name=f"{gr_attach.get_value('file_name')}",
            attachment_type=f"{gr_attach.get_value('content_type')}",
            attachment_size=f"{gr_attach.get_value('size_bytes')}",
            attachment_link=f"https://{self.instance}.service-now.com/sys_attachment.do?sys_id={gr_attach.get_value('sys_id')}",
            error=error or "",  # Ensure error is a string
        )

    def handle_attachment(self, gr_attach: GlideRecord, kb_number: str) -> dict:
        """
        Process a single attachment GlideRecord and return its info dict.
        """
        if not hasattr(gr_attach, "file_name") or not hasattr(
            gr_attach, "content_type"
        ):
            self.logger.error(
                "Invalid GlideRecord for attachment, missing required fields."
            )
            return {}

        attachment_id = f"{gr_attach.get_value('sys_id')}"
        size_bytes = int(f"{gr_attach.get_value('size_bytes')}")
        file_name = f"{gr_attach.get_value('file_name')}"
        content_type = f"{gr_attach.get_value('content_type')}"

        self.logger.info(f"Processing attachment {file_name}")
        self.observer.notify(
            self.get_attachment_event(
                event_name=EventName.ATTACHMENT_PROCESSING_STARTED,
                page_id=kb_number,
                gr_attach=gr_attach,
            )
        )

        if self.process_attachment_callback:
            can_process, message = self.process_attachment_callback(
                content_type, size_bytes, file_name
            )
            if not can_process:
                self.observer.notify(
                    self.get_attachment_event(
                        event_name=EventName.ATTACHMENT_SKIPPED,
                        page_id=kb_number,
                        gr_attach=gr_attach,
                        error=message,
                    )
                )
                self.logger.info(f"Skipping attachment {file_name}: {message}")
                return {}

        try:
            res: requests.Response = self._download_attachment_content(gr_attach.sys_id)
            if not res or not getattr(res, "ok", False):
                self.logger.error(
                    f"Failed to download attachment content for {file_name}"
                )
                return {}
            else:
                file_content = res.content

            file_type = self.get_File_type(file_name)
            try:
                markdown_text = self.custom_parser_manager.process_with_custom_parser(
                    file_type, file_content, file_name.split(".")[-1]
                )
            except ValueError as e:
                self.logger.error(
                    f"Error processing attachment {file_name} with custom parser: {e}"
                )
                self.observer.notify(
                    self.get_attachment_event(
                        event_name=EventName.ATTACHMENT_FAILED,
                        page_id=kb_number,
                        gr_attach=gr_attach,
                        error=str(e),
                    )
                )
                if self.fail_on_error:
                    raise
                return {}  # Skip this attachment if custom parser fails

            self.logger.debug(markdown_text)

            self.observer.notify(
                self.get_attachment_event(
                    event_name=EventName.ATTACHMENT_PROCESSED,
                    page_id=kb_number,
                    gr_attach=gr_attach,
                )
            )
            return {
                "file_name": file_name,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "markdown_text": markdown_text,
                "sys_id": gr_attach.sys_id,
            }
        except Exception as e:
            self.logger.error(f"Error processing attachment {file_name}: {e}")
            self.observer.notify(
                self.get_attachment_event(
                    event_name=EventName.ATTACHMENT_FAILED,
                    page_id=kb_number,
                    gr_attach=gr_attach,
                    error=str(e),
                )
            )
            return {}

    def handle_attachments(self, sys_id: str, kb_number: str) -> list:
        """
        Download all attachments for a given KB article sys_id. Returns a list of attachment info dicts.
        """
        attachments = []
        try:
            gr_attach = self.pysnc_client.GlideRecord("sys_attachment")
            gr_attach.add_query("table_sys_id", sys_id)
            gr_attach.add_query("table_name", self.kb_table)
            gr_attach.query()
            while gr_attach.next():
                attachment_info = self.handle_attachment(gr_attach, kb_number)
                if "markdown_text" in attachment_info:
                    attachments.append(attachment_info)
        except Exception as e:
            self.logger.error(f"Error downloading attachments: {e}")
        return attachments

    def load_data(
        self, article_sys_id: Optional[str] = None, numbers: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load a KB article by sys_id or number using PySNC. Returns a list with one Document.
        """
        article_sys_id = article_sys_id or ""
        return self.get_documents(
            article_sys_id=article_sys_id, numbers=numbers, status="Published"
        )

    def get_File_type(self, file_name: str) -> FileType:
        """
        Determine the file type based on the file name extension.
        """
        ext = os.path.splitext(file_name)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".gif"]:
            return FileType.IMAGE
        elif ext in [".pdf"]:
            return FileType.PDF
        elif ext in [".txt"]:
            return FileType.TEXT
        elif ext in [".csv"]:
            return FileType.CSV
        elif ext in [".html"]:
            return FileType.HTML
        elif ext in [".docx"]:
            return FileType.DOCUMENT
        elif ext in [".xlsx"]:
            return FileType.SPREADSHEET
        elif ext in [".pptx"]:
            return FileType.PRESENTATION
        elif ext in [".md"]:
            return FileType.MARKDOWN
        else:
            return FileType.UNKNOWN

    def _download_attachment_content(self, sys_id: str) -> Optional[bytes]:
        """
        Download attachment content using PySNC's attachment.get_file method.
        """
        try:
            if hasattr(self.pysnc_client, "attachment_api") and hasattr(
                self.pysnc_client.attachment_api, "get_file"
            ):
                return self.pysnc_client.attachment_api.get_file(sys_id)
            else:
                self.logger.error(
                    "self.pysnc_client.attachment_api.get_file is not available. Please check your PySNC version."
                )
                return None
        except Exception as e:
            self.logger.error(f"Attachment download failed for {sys_id}: {e}")
            return None
