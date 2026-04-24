"""Confluence reader."""

import logging
import os
import tempfile
from typing import Callable, Dict, List, Optional
from urllib.parse import unquote

from llama_index_instrumentation import DispatcherSpanMixin, get_dispatcher
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.utils import get_tqdm_iterable
from retrying import retry

from .default_parsers import get_default_parsers
from .event import (
    AttachmentFailedEvent,
    AttachmentProcessedEvent,
    AttachmentProcessingStartedEvent,
    AttachmentSkippedEvent,
    FileType,
    PageDataFetchCompletedEvent,
    PageDataFetchStartedEvent,
    PageFailedEvent,
    PageSkippedEvent,
    TotalPagesToProcessEvent,
)

CONFLUENCE_API_TOKEN = "CONFLUENCE_API_TOKEN"
CONFLUENCE_PASSWORD = "CONFLUENCE_PASSWORD"
CONFLUENCE_USERNAME = "CONFLUENCE_USERNAME"

internal_logger = logging.getLogger(__name__)
dispatcher = get_dispatcher(__name__)


class CustomParserManager:
    def __init__(
        self, custom_parsers: Optional[Dict[FileType, BaseReader]], custom_folder: str
    ):
        self.custom_parsers = custom_parsers or {}
        self.custom_folder = custom_folder

    def process_with_custom_parser(
        self, file_type: FileType, file_content: bytes, extension: str
    ) -> tuple[str, dict]:
        if file_type not in self.custom_parsers:
            return "", {}

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=f".{extension}",
            dir=self.custom_folder,
            delete=False,
        ) as f:
            f.write(file_content)
            custom_file_path = f.name
        try:
            docs = self.custom_parsers[file_type].load_data(file_path=custom_file_path)
            text = "\n".join(doc.text for doc in docs) + "\n"
            # TODO: If a parser returns multiple Documents each with different metadata, how should those be merged together?
            metadata = docs[0].metadata or {} if docs else {}
        finally:
            try:
                os.unlink(custom_file_path)
            except OSError:
                pass
        return text, metadata


class ConfluenceReader(BaseReader, DispatcherSpanMixin):
    """
    Confluence reader.

    Reads Confluence pages and optional attachments selected by one query method.

    Authentication precedence:
        1. ``oauth2``
        2. ``api_token``
        3. ``cookies``
        4. ``user_name`` and ``password``
        5. Environment variables:
           ``CONFLUENCE_API_TOKEN`` or ``CONFLUENCE_USERNAME`` +
           ``CONFLUENCE_PASSWORD``

    If none of the above are provided, initialization raises ``ValueError``.

    Query constraints:
        - ``load_data`` requires exactly one of ``space_key``, ``page_ids``,
          ``label``, ``folder_id``, or ``cql``.
        - ``include_children`` is only valid with ``page_ids``.
        - ``page_status`` is only valid with ``space_key``.
        - ``start`` and ``cursor`` are mutually exclusive.
        - ``cursor`` is not valid with ``space_key``.

    Args:
        base_url (str): Base URL for the Confluence instance, usually ending
            with ``/wiki`` for Confluence Cloud.
        oauth2 (Optional[dict]): Atlassian OAuth 2.0 configuration. Minimum
            expected keys include ``client_id`` and ``token`` where ``token``
            contains at least ``access_token`` and ``token_type``.
        cloud (bool): Whether to connect to Confluence Cloud (`True`) or a
            self-hosted instance (`False`).
        api_token (Optional[str]): Confluence API token for token-based auth.
            See: https://confluence.atlassian.com/cloud/api-tokens-938839638.html
        cookies (Optional[dict]): Cookie-based auth payload accepted by
            ``atlassian.Confluence``.
        user_name (Optional[str]): Username for basic auth. Must be used with
            ``password``.
        password (Optional[str]): Password for basic auth. Must be used with
            ``user_name``.
        client_args (Optional[dict]): Additional keyword arguments passed to
            ``atlassian.Confluence`` (for example,
            ``{"backoff_and_retry": True}``).
        custom_parsers (Optional[Dict[FileType, BaseReader]]): Per-file-type
            parser overrides. Entries are merged with built-in parsers and
            override defaults for matching ``FileType`` values.
        process_attachment_callback (Optional[Callable[[str, int, str],
            tuple[bool, str]]]): Callback to decide whether to process an
            attachment. Receives ``(media_type, file_size, attachment_title)``
            and returns ``(should_process, reason)``.
        process_document_callback (Optional[Callable[[str], bool]]): Callback
            to decide whether to process a page. Receives ``page_id`` and
            returns ``True`` to process or ``False`` to skip.
        custom_folder (Optional[str]): Folder used for temporary parser files.
            Defaults to the current working directory.
        logger (Optional[logging.Logger]): Custom logger instance. If not
            provided, uses the module logger.
        fail_on_error (bool): Error policy for page and attachment processing.
            If ``True`` (default), raises on processing errors. If ``False``,
            logs warnings, emits failure events, and continues processing.
        verbose (bool): Whether to emit detailed log messages.

    Instrumentation Events:
        The reader emits LlamaIndex instrumentation events during page and
        attachment processing. Add event handlers to the dispatcher to capture
        these events.

        Available events:
        - ``TotalPagesToProcessEvent``: ``total_pages``. Emitted when the total number of pages to process is determined.
        - ``PageDataFetchStartedEvent``: ``page_id``. Emitted when processing of a page begins.
        - ``PageDataFetchCompletedEvent``: ``page_id``, ``document``. Emitted when a page is successfully processed.
        - ``PageFailedEvent``: ``page_id``, ``error``. Emitted when page processing fails.
        - ``PageSkippedEvent``: ``page_id``. Emitted when a page is skipped due to callback decision.
        - ``AttachmentProcessingStartedEvent``: ``page_id``, ``attachment_id``,
          ``attachment_name``, ``attachment_type``, ``attachment_size``,
          ``attachment_link``. Emitted when attachment processing begins.
        - ``AttachmentProcessedEvent``: ``page_id``, ``attachment_id``,
          ``attachment_name``, ``attachment_type``, ``attachment_size``,
          ``attachment_link``. Emitted when an attachment is successfully processed.
        - ``AttachmentSkippedEvent``: ``page_id``, ``attachment_id``,
          ``attachment_name``, ``attachment_type``, ``attachment_size``,
          ``attachment_link``, ``reason``. Emitted when an attachment is skipped.
        - ``AttachmentFailedEvent``: ``page_id``, ``attachment_id``,
          ``attachment_name``, ``attachment_type``, ``attachment_size``,
          ``attachment_link``, ``error``. Emitted when attachment processing fails.

        Unsupported attachment media types are skipped and emit
        ``AttachmentSkippedEvent``.

        To listen to events, add an event handler to the dispatcher:
        ```python
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.instrumentation.event_handlers import BaseEventHandler

        class MyEventHandler(BaseEventHandler):
            def handle(self, event):
                print(f"Event: {event.class_name()}")

        dispatcher = get_dispatcher(__name__)
        dispatcher.add_event_handler(MyEventHandler())
        ```

    References:
        - https://atlassian-python-api.readthedocs.io/index.html
        - https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/

    """

    def __init__(
        self,
        base_url: str,
        *,
        oauth2: Optional[Dict] = None,
        cloud: bool = True,
        api_token: Optional[str] = None,
        cookies: Optional[dict] = None,
        user_name: Optional[str] = None,
        password: Optional[str] = None,
        client_args: Optional[dict] = None,
        custom_parsers: Optional[Dict[FileType, BaseReader]] = None,
        process_attachment_callback: Optional[
            Callable[[str, int, str], tuple[bool, str]]
        ] = None,
        process_document_callback: Optional[Callable[[str], bool]] = None,
        custom_folder: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        fail_on_error: bool = True,
        verbose: bool = False,
    ) -> None:
        if base_url is None:
            raise ValueError("Must provide `base_url`")

        self.base_url = base_url

        self.custom_parser_manager = CustomParserManager(
            {**get_default_parsers(), **(custom_parsers or {})},
            custom_folder or os.getcwd(),
        )

        self.logger = logger or internal_logger
        self.process_attachment_callback = process_attachment_callback
        self.process_document_callback = process_document_callback
        self.fail_on_error = fail_on_error
        self.verbose = verbose

        try:
            from atlassian import Confluence
        except ImportError:
            raise ImportError(
                "`atlassian` package not found, please run `pip install atlassian-python-api`"
            )

        if client_args is None:
            client_args = {}
        if oauth2:
            self.confluence = Confluence(
                url=base_url, oauth2=oauth2, cloud=cloud, **client_args
            )
        else:
            if api_token is not None:
                self.confluence = Confluence(
                    url=base_url, token=api_token, cloud=cloud, **client_args
                )
            elif cookies is not None:
                self.confluence = Confluence(
                    url=base_url, cookies=cookies, cloud=cloud, **client_args
                )
            elif user_name is not None and password is not None:
                self.confluence = Confluence(
                    url=base_url,
                    username=user_name,
                    password=password,
                    cloud=cloud,
                    **client_args,
                )
            else:
                api_token = os.getenv(CONFLUENCE_API_TOKEN)
                if api_token is not None:
                    self.confluence = Confluence(
                        url=base_url, token=api_token, cloud=cloud, **client_args
                    )
                else:
                    user_name = os.getenv(CONFLUENCE_USERNAME)
                    password = os.getenv(CONFLUENCE_PASSWORD)
                    if user_name is not None and password is not None:
                        self.confluence = Confluence(
                            url=base_url,
                            username=user_name,
                            password=password,
                            cloud=cloud,
                            **client_args,
                        )
                    else:
                        raise ValueError(
                            "Must set one of environment variables `CONFLUENCE_API_KEY`, or"
                            " `CONFLUENCE_USERNAME` and `CONFLUENCE_PASSWORD`, if oauth2, or"
                            " api_token, or user_name and password parameters are not provided"
                        )

        self._next_cursor = None

    def _format_attachment_header(self, attachment: dict) -> str:
        """Formats the attachment title as a markdown header."""
        return f"# {attachment['title']}\n"

    @dispatcher.span
    def load_data(
        self,
        space_key: Optional[str] = None,
        page_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        page_status: Optional[str] = None,
        label: Optional[str] = None,
        cql: Optional[str] = None,
        include_attachments=False,
        include_children=False,
        start: Optional[int] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        max_num_results: Optional[int] = None,
    ) -> List[Document]:
        """
        Load Confluence pages from Confluence, specifying one of five mutually exclusive methods:
        `space_key`, `page_ids`, `label`, `folder_id` or `cql`
        (Confluence Query Language https://developer.atlassian.com/cloud/confluence/advanced-searching-using-cql/ ).

        Args:
            space_key (str): Confluence space key, eg 'DS'
            page_ids (list): List of page ids, eg ['123456', '123457']
            folder_id (str): Confluence folder id, eg '1234567890'
            page_status (str): Page status, one of None (all statuses), 'current', 'draft', 'archived'.  Only compatible with space_key.
            label (str): Confluence label, eg 'my-label'
            cql (str): Confluence Query Language query, eg 'label="my-label"'
            include_attachments (bool): If True, include attachments.
            include_children (bool): If True, do a DFS of the descendants of each page_id in `page_ids`.  Only compatible with `page_ids`.
            start (int): Skips over the first n elements. Used only with space_key
            cursor (str): Skips to the cursor. Used with cql and label, set when the max limit has been hit for cql based search
            limit (int): Deprecated, use `max_num_results` instead.
            max_num_results (int): Maximum number of results to return.  If None, return all results.  Requests are made in batches to achieve the desired number of results.

        """
        num_space_key_parameter = 1 if space_key else 0
        num_page_ids_parameter = 1 if page_ids is not None else 0
        num_label_parameter = 1 if label else 0
        num_cql_parameter = 1 if cql else 0
        num_folder_id_parameter = 1 if folder_id else 0
        if (
            num_space_key_parameter
            + num_page_ids_parameter
            + num_label_parameter
            + num_cql_parameter
            + num_folder_id_parameter
            != 1
        ):
            raise ValueError(
                "Must specify exactly one among `space_key`, `page_ids`, `label`, `folder_id`, `cql` parameters."
            )

        if cursor and start:
            raise ValueError("Must not specify `start` when `cursor` is specified")

        if space_key and cursor:
            raise ValueError("Must not specify `cursor` when `space_key` is specified")

        if page_status and not space_key:
            raise ValueError(
                "Must specify `space_key` when `page_status` is specified."
            )

        if include_children and not page_ids:
            raise ValueError(
                "Must specify `page_ids` when `include_children` is specified."
            )

        if limit is not None:
            max_num_results = limit
            self.logger.warning(
                "`limit` is deprecated and no longer relates to the Confluence server's"
                " API limits.  If you wish to limit the number of returned results"
                " please use `max_num_results` instead."
            )

        if not start:
            start = 0

        pages: List = []
        if space_key:
            pages.extend(
                self._get_data_with_paging(
                    self.confluence.get_all_pages_from_space,
                    start=start,
                    max_num_results=max_num_results,
                    space=space_key,
                    status=page_status,
                    expand="body.export_view.value",
                    content_type="page",
                )
            )
        elif label:
            pages.extend(
                self._get_cql_data_with_paging(
                    start=start,
                    cursor=cursor,
                    cql=f'type="page" AND label="{label}"',
                    max_num_results=max_num_results,
                    expand="body.export_view.value",
                )
            )
        elif cql:
            pages.extend(
                self._get_cql_data_with_paging(
                    start=start,
                    cursor=cursor,
                    cql=cql,
                    max_num_results=max_num_results,
                    expand="body.export_view.value",
                )
            )
        elif page_ids:
            if include_children:
                dfs_page_ids = []
                max_num_remaining = max_num_results
                for page_id in page_ids:
                    current_dfs_page_ids = self._dfs_page_ids(
                        page_id,
                        type="page",
                        max_num_results=max_num_remaining,
                    )
                    dfs_page_ids.extend(current_dfs_page_ids)
                    if max_num_results is not None:
                        max_num_remaining -= len(current_dfs_page_ids)
                        if max_num_remaining <= 0:
                            break
                page_ids = dfs_page_ids
            page_ids_with_progress = get_tqdm_iterable(
                page_ids[:max_num_results] if max_num_results is not None else page_ids,
                self.verbose,
                "Downloading Confluence pages",
            )
            for page_id in page_ids_with_progress:
                try:
                    pages.append(
                        self._get_data_with_retry(
                            self.confluence.get_page_by_id,
                            page_id=page_id,
                            expand="body.export_view.value",
                        )
                    )
                except Exception:
                    if self.fail_on_error:
                        self.logger.error(f"Failed to fetch page with id {page_id}")
                        raise
                    else:
                        self.logger.warning(
                            f"Failed to fetch page with id {page_id}. Skipping this page."
                        )
        elif folder_id:
            # Fetch all folders in the folder
            max_num_remaining = max_num_results
            page_ids = self._dfs_page_ids(
                folder_id,
                type="folder",
                max_num_results=max_num_remaining,
            )
            page_ids_with_progress = get_tqdm_iterable(
                page_ids[:max_num_results] if max_num_results is not None else page_ids,
                self.verbose,
                "Downloading Confluence pages",
            )
            for page_id in page_ids_with_progress:
                try:
                    pages.append(
                        self._get_data_with_retry(
                            self.confluence.get_page_by_id,
                            page_id=page_id,
                            expand="body.export_view.value",
                        )
                    )
                except Exception:
                    if self.fail_on_error:
                        self.logger.error(f"Failed to fetch page with id {page_id}")
                        raise
                    else:
                        self.logger.warning(
                            f"Failed to fetch page with id {page_id}. Skipping this page."
                        )
        docs = []

        if pages:
            dispatcher.event(TotalPagesToProcessEvent(total_pages=len(pages)))

        for page in pages:
            try:
                doc = self.process_page(page, include_attachments)
                if doc:
                    docs.append(doc)
            except Exception as e:
                self.logger.error(f"Error processing page {page['id']}: {e}")
                dispatcher.event(PageFailedEvent(page_id=page["id"], error=str(e)))
                if self.fail_on_error:
                    raise
                else:
                    self.logger.warning(
                        f"Failed to process page {page['id']}: {e}. Skipping this page."
                    )
        return docs

    def _dfs_page_ids(self, id, type="page", max_num_results=None):
        ret = []

        max_num_remaining = max_num_results
        if type == "page":
            ret.append(id)
            if max_num_remaining is not None:
                max_num_remaining -= 1
                if max_num_remaining < 0:
                    return ret

        # Fetch both page and folder children with their types
        child_items = [
            (child_id, "page")
            for child_id in self._get_data_with_paging(
                self.confluence.get_child_id_list,
                page_id=id,
                type="page",
                max_num_results=max_num_remaining,
            )
        ]

        if self.confluence.cloud:
            child_items.extend(
                [
                    (child_id, "folder")
                    for child_id in self._get_data_with_paging(
                        self.confluence.get_child_id_list,
                        page_id=id,
                        type="folder",
                        max_num_results=max_num_remaining,
                    )
                ]
            )

        for child_id, child_type in child_items:
            if max_num_remaining is not None and max_num_remaining <= 0:
                break

            dfs_ids = self._dfs_page_ids(
                child_id, type=child_type, max_num_results=max_num_remaining
            )
            ret.extend(dfs_ids)

            if max_num_remaining is not None:
                max_num_remaining -= len(dfs_ids)
                if max_num_remaining <= 0:
                    break

        return ret

    def _get_data_with_paging(
        self, paged_function, start=0, max_num_results=50, **kwargs
    ):
        max_num_remaining = max_num_results
        ret = []
        while True:
            results = self._get_data_with_retry(
                paged_function, start=start, limit=max_num_remaining, **kwargs
            )
            ret.extend(results)
            if (
                len(results) == 0
                or max_num_results is not None
                and len(results) >= max_num_remaining
            ):
                break

            start += len(results)
            if max_num_remaining is not None:
                max_num_remaining -= len(results)
        return ret

    def _get_cql_data_with_paging(
        self,
        cql,
        start=0,
        cursor=None,
        max_num_results=50,
        expand="body.export_view.value",
    ):
        max_num_remaining = max_num_results
        ret = []
        params = {"cql": cql, "start": start, "expand": expand}
        if cursor:
            params["cursor"] = unquote(cursor)

        if max_num_results is not None:
            params["limit"] = max_num_remaining
        while True:
            results = self._get_data_with_retry(
                self.confluence.get, path="rest/api/content/search", params=params
            )
            ret.extend(results["results"])

            params["start"] += len(results["results"])

            next_url = (
                results["_links"]["next"] if "next" in results["_links"] else None
            )
            if not next_url:
                self._next_cursor = None
                break

            if "cursor=" in next_url:  # On confluence Server this is not set
                cursor = next_url.split("cursor=")[1].split("&")[0]
                params["cursor"] = unquote(cursor)

            if max_num_results is not None:
                params["limit"] -= len(results["results"])
                if params["limit"] <= 0:
                    self._next_cursor = cursor
                    break

        return ret

    def get_next_cursor(self):
        """
        Returns: The last set cursor from a cql based search.
        """
        return self._next_cursor

    @retry(stop_max_attempt_number=1, wait_fixed=4)
    def _get_data_with_retry(self, function, **kwargs):
        return function(**kwargs)

    @dispatcher.span
    def process_page(self, page, include_attachments):
        self.logger.info(
            f"Processing {self.base_url}{page['_links']['webui']} ({page['title']})"
        )

        if self.process_document_callback:
            should_process = self.process_document_callback(page["id"])
            if not should_process:
                self.logger.info(
                    f"Skipping page {page['id']} based on callback decision."
                )
                dispatcher.event(PageSkippedEvent(page_id=page["id"]))
                return None

        dispatcher.event(PageDataFetchStartedEvent(page_id=page["id"]))

        attachment_texts = []
        if include_attachments:
            attachment_texts = self.process_attachment(page["id"])

        html_bytes = page["body"]["export_view"]["value"].encode("utf-8")
        page_body, parser_metadata = (
            self.custom_parser_manager.process_with_custom_parser(
                FileType.PAGE_HTML, html_bytes, "html"
            )
        )
        text = page_body + "".join(attachment_texts)

        default_metadata = {
            "title": page["title"],
            "page_id": page["id"],
            "status": page["status"],
            "url": self.base_url + page["_links"]["webui"],
        }
        doc = Document(
            text=text,
            doc_id=page["id"],
            metadata={**parser_metadata, **default_metadata},
        )
        dispatcher.event(PageDataFetchCompletedEvent(page_id=page["id"], document=doc))
        return doc

    @dispatcher.span
    def process_attachment(self, page_id):
        # depending on setup you may also need to set the correct path for poppler and tesseract
        attachments = self.confluence.get_attachments_from_content(page_id)["results"]
        texts = []
        if not attachments:
            return texts

        for attachment in attachments:
            self.logger.info("Processing attachment " + attachment["title"])
            dispatcher.event(
                AttachmentProcessingStartedEvent(
                    page_id=page_id,
                    attachment_id=attachment["id"],
                    attachment_name=attachment["title"],
                    attachment_type=attachment["metadata"]["mediaType"],
                    attachment_size=attachment["extensions"]["fileSize"],
                    attachment_link=attachment["_links"]["webui"],
                )
            )

            if self.process_attachment_callback:
                should_process, reason = self.process_attachment_callback(
                    attachment["metadata"]["mediaType"],
                    attachment["extensions"]["fileSize"],
                    attachment["title"],
                )
                if not should_process:
                    self.logger.info(
                        f"Skipping attachment {attachment['title']} based on callback decision."
                    )
                    dispatcher.event(
                        AttachmentSkippedEvent(
                            page_id=page_id,
                            attachment_id=attachment["id"],
                            attachment_name=attachment["title"],
                            attachment_type=attachment["metadata"]["mediaType"],
                            attachment_size=attachment["extensions"]["fileSize"],
                            attachment_link=attachment["_links"]["webui"],
                            reason=reason,
                        )
                    )
                    continue

            try:
                media_type = attachment["metadata"]["mediaType"]
                absolute_url = self.base_url + attachment["_links"]["download"]
                title = self._format_attachment_header(attachment)
                if media_type == "application/pdf":
                    self.logger.info("Processing PDF attachment " + absolute_url)
                    text = title + self.process_pdf(absolute_url)
                elif (
                    media_type == "image/png"
                    or media_type == "image/jpg"
                    or media_type == "image/jpeg"
                    or media_type == "image/webp"
                ):
                    self.logger.info("Processing image attachment " + absolute_url)
                    text = title + self.process_image(absolute_url)
                elif (
                    media_type
                    == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ):
                    self.logger.info(
                        "Processing Word document attachment " + absolute_url
                    )
                    text = title + self.process_doc(absolute_url)
                elif (
                    media_type == "application/vnd.ms-excel"
                    or media_type
                    == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    or media_type == "application/vnd.ms-excel.sheet.macroenabled.12"
                ):
                    if attachment["title"].endswith(".csv") or absolute_url.endswith(
                        ".csv"
                    ):
                        self.logger.info("Processing CSV attachment " + absolute_url)
                        text = title + self.process_csv(absolute_url)
                    else:
                        self.logger.info("Processing XLS attachment " + absolute_url)
                        text = title + self.process_xls(absolute_url)
                elif (
                    media_type
                    == "application/vnd.ms-excel.sheet.binary.macroenabled.12"
                ):
                    self.logger.info("Processing XLSB attachment " + absolute_url)
                    text = title + self.process_xlsb(absolute_url)
                elif media_type == "text/csv":
                    self.logger.info("Processing CSV attachment " + absolute_url)
                    text = title + self.process_csv(absolute_url)
                elif media_type == "application/vnd.ms-outlook":
                    self.logger.info(
                        "Processing Outlook message attachment " + absolute_url
                    )
                    text = title + self.process_msg(absolute_url)
                elif media_type == "text/html":
                    self.logger.info("  Processing HTML attachment " + absolute_url)
                    text = title + self.process_html(absolute_url)
                elif media_type == "text/plain":
                    if attachment["title"].endswith(".csv") or absolute_url.endswith(
                        ".csv"
                    ):
                        self.logger.info("Processing CSV attachment " + absolute_url)
                        text = title + self.process_csv(absolute_url)
                    else:
                        self.logger.info("Processing Text attachment " + absolute_url)
                        text = title + self.process_txt(absolute_url)
                elif media_type == "text/markdown" or absolute_url.endswith(
                    (".md", ".mdx")
                ):
                    self.logger.info("Processing Markdown attachment " + absolute_url)
                    text = title + self.process_txt(absolute_url)
                elif media_type == "image/svg+xml":
                    self.logger.info("Processing SVG attachment " + absolute_url)
                    text = title + self.process_svg(absolute_url)
                elif (
                    media_type
                    == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    or media_type
                    == "application/vnd.ms-powerpoint.presentation.macroenabled.12"
                ):
                    self.logger.info(
                        "Processing PowerPoint attachment "
                        + absolute_url
                        + " ("
                        + media_type
                        + ")"
                    )
                    text = title + self.process_ppt(absolute_url)
                elif media_type == "binary/octet-stream" and (
                    attachment["title"].strip().endswith(".md")
                    or attachment["title"].strip().endswith(".mdx")
                ):
                    self.logger.info("Processing Markdown attachment " + absolute_url)
                    text = title + self.process_txt(absolute_url)
                else:
                    self.logger.info(
                        f"Skipping unsupported attachment {absolute_url} of media_type {media_type}"
                    )
                    dispatcher.event(
                        AttachmentSkippedEvent(
                            page_id=page_id,
                            attachment_id=attachment["id"],
                            attachment_name=attachment["title"],
                            attachment_type=attachment["metadata"]["mediaType"],
                            attachment_size=attachment["extensions"]["fileSize"],
                            attachment_link=attachment["_links"]["webui"],
                            reason="Unsupported media type",
                        )
                    )
                    continue
                texts.append(text)
                dispatcher.event(
                    AttachmentProcessedEvent(
                        page_id=page_id,
                        attachment_id=attachment["id"],
                        attachment_name=attachment["title"],
                        attachment_type=attachment["metadata"]["mediaType"],
                        attachment_size=attachment["extensions"]["fileSize"],
                        attachment_link=attachment["_links"]["webui"],
                    )
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to process attachment {attachment['title']}: {e}"
                )
                dispatcher.event(
                    AttachmentFailedEvent(
                        page_id=page_id,
                        attachment_id=attachment["id"],
                        attachment_name=attachment["title"],
                        attachment_type=attachment["metadata"]["mediaType"],
                        attachment_size=attachment["extensions"]["fileSize"],
                        attachment_link=attachment["_links"]["webui"],
                        error=str(e),
                    )
                )
                # Enforce fail_on_error parameter: if True, re-raise exception to stop processing
                if self.fail_on_error:
                    raise
                else:
                    self.logger.warning(
                        f"Failed to process attachment {attachment['title']}: {e}. Skipping this attachment."
                    )

        return texts

    def process_pdf(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching PDF attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.PDF, response.content, "pdf"
        )
        return text

    def process_html(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching HTML attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.HTML, response.content, "html"
        )
        return text

    def process_txt(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching TXT attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.TEXT, response.content, "txt"
        )
        return text

    def process_msg(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching MSG attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.MSG, response.content, "msg"
        )
        return text

    def process_image(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching image attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.IMAGE, response.content, "png"
        )
        return text

    def process_doc(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching document attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.DOCUMENT, response.content, "docx"
        )
        return text

    def process_ppt(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching PPT attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.PRESENTATION, response.content, "pptx"
        )
        return text

    def process_xls(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching XLS attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.SPREADSHEET, response.content, "xlsx"
        )
        return text

    def process_xlsb(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching XLSB attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.XLSB, response.content, "xlsb"
        )
        return text

    def process_csv(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching CSV attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.CSV, response.content, "csv"
        )
        return text

    def process_svg(self, link) -> str:
        response = self.confluence.request(path=link, absolute=True)
        if response.status_code != 200 or not response.content:
            self.logger.error(
                f"Error fetching SVG attachment at {link}: HTTP status code {response.status_code}."
            )
            return ""
        text, _ = self.custom_parser_manager.process_with_custom_parser(
            FileType.SVG, response.content, "svg"
        )
        return text
