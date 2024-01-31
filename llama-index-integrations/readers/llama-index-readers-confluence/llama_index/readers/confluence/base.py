"""Confluence reader."""
import logging
import os
from typing import Dict, List, Optional
from urllib.parse import unquote

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from retrying import retry

CONFLUENCE_API_TOKEN = "CONFLUENCE_API_TOKEN"
CONFLUENCE_PASSWORD = "CONFLUENCE_PASSWORD"
CONFLUENCE_USERNAME = "CONFLUENCE_USERNAME"

logger = logging.getLogger(__name__)


class ConfluenceReader(BaseReader):
    """Confluence reader.

    Reads a set of confluence pages given a space key and optionally a list of page ids

    For more on OAuth login, checkout:
        - https://atlassian-python-api.readthedocs.io/index.html
        - https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/

    Args:
        oauth2 (dict): Atlassian OAuth 2.0, minimum fields are `client_id` and `token`, where `token` is a dict and must at least contain "access_token" and "token_type".
        base_url (str): 'base_url' for confluence cloud instance, this is suffixed with '/wiki', eg 'https://yoursite.atlassian.com/wiki'
        cloud (bool): connecting to Confluence Cloud or self-hosted instance

    """

    def __init__(
        self, base_url: str = None, oauth2: Optional[Dict] = None, cloud: bool = True
    ) -> None:
        if base_url is None:
            raise ValueError("Must provide `base_url`")

        self.base_url = base_url

        try:
            from atlassian import Confluence
        except ImportError:
            raise ImportError(
                "`atlassian` package not found, please run `pip install"
                " atlassian-python-api`"
            )
        self.confluence: Confluence = None
        if oauth2:
            self.confluence = Confluence(url=base_url, oauth2=oauth2, cloud=cloud)
        else:
            api_token = os.getenv(CONFLUENCE_API_TOKEN)
            if api_token is not None:
                self.confluence = Confluence(url=base_url, token=api_token, cloud=cloud)
            else:
                user_name = os.getenv(CONFLUENCE_USERNAME)
                if user_name is None:
                    raise ValueError(
                        "Must set environment variable `CONFLUENCE_USERNAME` if oauth,"
                        " oauth2, or `CONFLUENCE_API_TOKEN` are not provided."
                    )
                password = os.getenv(CONFLUENCE_PASSWORD)
                if password is None:
                    raise ValueError(
                        "Must set environment variable `CONFLUENCE_PASSWORD` if oauth,"
                        " oauth2, or `CONFLUENCE_API_TOKEN` are not provided."
                    )
                self.confluence = Confluence(
                    url=base_url, username=user_name, password=password, cloud=cloud
                )

        self._next_cursor = None

    def load_data(
        self,
        space_key: Optional[str] = None,
        page_ids: Optional[List[str]] = None,
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
        """Load Confluence pages from Confluence, specifying by one of four mutually exclusive methods:
        `space_key`, `page_ids`, `label`, or `cql`
        (Confluence Query Language https://developer.atlassian.com/cloud/confluence/advanced-searching-using-cql/ ).

        Args:
            space_key (str): Confluence space key, eg 'DS'
            page_ids (list): List of page ids, eg ['123456', '123457']
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
        if (
            num_space_key_parameter
            + num_page_ids_parameter
            + num_label_parameter
            + num_cql_parameter
            != 1
        ):
            raise ValueError(
                "Must specify exactly one among `space_key`, `page_ids`, `label`, `cql`"
                " parameters."
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
            logger.warning(
                "`limit` is deprecated and no longer relates to the Confluence server's"
                " API limits.  If you wish to limit the number of returned results"
                " please use `max_num_results` instead."
            )

        try:
            import html2text  # type: ignore
        except ImportError:
            raise ImportError(
                "`html2text` package not found, please run `pip install html2text`"
            )

        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.ignore_images = True

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
                        page_id, max_num_remaining
                    )
                    dfs_page_ids.extend(current_dfs_page_ids)
                    if max_num_results is not None:
                        max_num_remaining -= len(current_dfs_page_ids)
                        if max_num_remaining <= 0:
                            break
                page_ids = dfs_page_ids
            for page_id in (
                page_ids[:max_num_results] if max_num_results is not None else page_ids
            ):
                pages.append(
                    self._get_data_with_retry(
                        self.confluence.get_page_by_id,
                        page_id=page_id,
                        expand="body.export_view.value",
                    )
                )

        docs = []
        for page in pages:
            doc = self.process_page(page, include_attachments, text_maker)
            docs.append(doc)

        return docs

    def _dfs_page_ids(self, page_id, max_num_results):
        ret = [page_id]
        max_num_remaining = (
            (max_num_results - 1) if max_num_results is not None else None
        )
        if max_num_results is not None and max_num_remaining <= 0:
            return ret

        child_page_ids = self._get_data_with_paging(
            self.confluence.get_child_id_list,
            page_id=page_id,
            type="page",
            max_num_results=max_num_remaining,
        )
        for child_page_id in child_page_ids:
            dfs_ids = self._dfs_page_ids(child_page_id, max_num_remaining)
            ret.extend(dfs_ids)
            if max_num_results is not None:
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
        Returns: The last set cursor from a cql based search
        """
        return self._next_cursor

    @retry(stop_max_attempt_number=1, wait_fixed=4)
    def _get_data_with_retry(self, function, **kwargs):
        return function(**kwargs)

    def process_page(self, page, include_attachments, text_maker):
        if include_attachments:
            attachment_texts = self.process_attachment(page["id"])
        else:
            attachment_texts = []
        text = text_maker.handle(page["body"]["export_view"]["value"]) + "".join(
            attachment_texts
        )
        return Document(
            text=text,
            doc_id=page["id"],
            extra_info={
                "title": page["title"],
                "page_id": page["id"],
                "status": page["status"],
                "url": self.base_url + page["_links"]["webui"],
            },
        )

    def process_attachment(self, page_id):
        try:
            pass
        except ImportError:
            raise ImportError(
                "`pytesseract` or `pdf2image` or `Pillow` package not found, please run"
                " `pip install pytesseract pdf2image Pillow`"
            )

        # depending on setup you may also need to set the correct path for poppler and tesseract
        attachments = self.confluence.get_attachments_from_content(page_id)["results"]
        texts = []
        for attachment in attachments:
            media_type = attachment["metadata"]["mediaType"]
            absolute_url = self.base_url + attachment["_links"]["download"]
            title = attachment["title"]
            if media_type == "application/pdf":
                text = title + self.process_pdf(absolute_url)
            elif (
                media_type == "image/png"
                or media_type == "image/jpg"
                or media_type == "image/jpeg"
            ):
                text = title + self.process_image(absolute_url)
            elif (
                media_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                text = title + self.process_doc(absolute_url)
            elif media_type == "application/vnd.ms-excel":
                text = title + self.process_xls(absolute_url)
            elif media_type == "image/svg+xml":
                text = title + self.process_svg(absolute_url)
            else:
                continue
            texts.append(text)

        return texts

    def process_pdf(self, link):
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_bytes  # type: ignore
        except ImportError:
            raise ImportError(
                "`pytesseract` or `pdf2image` package not found, please run `pip"
                " install pytesseract pdf2image`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        try:
            images = convert_from_bytes(response.content)
        except ValueError:
            return text

        for i, image in enumerate(images):
            image_text = pytesseract.image_to_string(image)
            text += f"Page {i + 1}:\n{image_text}\n\n"

        return text

    def process_image(self, link):
        try:
            from io import BytesIO  # type: ignore

            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            raise ImportError(
                "`pytesseract` or `Pillow` package not found, please run `pip install"
                " pytesseract Pillow`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        try:
            image = Image.open(BytesIO(response.content))
        except OSError:
            return text

        return pytesseract.image_to_string(image)

    def process_doc(self, link):
        try:
            from io import BytesIO  # type: ignore

            import docx2txt  # type: ignore
        except ImportError:
            raise ImportError(
                "`docx2txt` package not found, please run `pip install docx2txt`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        file_data = BytesIO(response.content)

        return docx2txt.process(file_data)

    def process_xls(self, link):
        try:
            import xlrd  # type: ignore
        except ImportError:
            raise ImportError("`xlrd` package not found, please run `pip install xlrd`")

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text

        workbook = xlrd.open_workbook(file_contents=response.content)
        for sheet in workbook.sheets():
            text += f"{sheet.name}:\n"
            for row in range(sheet.nrows):
                for col in range(sheet.ncols):
                    text += f"{sheet.cell_value(row, col)}\t"
                text += "\n"
            text += "\n"

        return text

    def process_svg(self, link):
        try:
            from io import BytesIO  # type: ignore

            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            from reportlab.graphics import renderPM  # type: ignore
            from svglib.svglib import svg2rlg  # type: ignore
        except ImportError:
            raise ImportError(
                "`pytesseract`, `Pillow`, or `svglib` package not found, please run"
                " `pip install pytesseract Pillow svglib`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text

        drawing = svg2rlg(BytesIO(response.content))

        img_data = BytesIO()
        renderPM.drawToFile(drawing, img_data, fmt="PNG")
        img_data.seek(0)
        image = Image.open(img_data)

        return pytesseract.image_to_string(image)


if __name__ == "__main__":
    reader = ConfluenceReader()
