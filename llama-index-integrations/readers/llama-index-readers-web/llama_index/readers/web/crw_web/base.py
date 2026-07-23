"""fastCRW Web Reader."""

from typing import Any, List, Optional, Dict, Callable

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class CrwWebReader(BasePydanticReader):
    """
    turn a url to llm accessible markdown with `fastCRW`.

    fastCRW is a Firecrawl-compatible web scraper that ships as a single
    binary. It can run against the managed cloud (default
    `https://fastcrw.com/api`) or a self-hosted server via `api_url`.

    Args:
        api_key (Optional[str]):
            The fastCRW API key. Optional: the underlying client reads
            `CRW_API_KEY` from the environment, and a self-hosted server may
            require no auth at all.
        api_url (Optional[str]):
            Optional base URL for a self-hosted fastCRW deployment. Defaults
            to the managed cloud when omitted.
        mode (Optional[str]):
            The mode to run the loader in. Default is "crawl".
            Options include "scrape" (single url),
            "crawl" (all accessible sub pages),
            "map" (map all accessible sub pages), and
            "search" (search for content).
        params (Optional[dict]): The parameters to pass to the fastCRW API.

    Examples include crawlerOptions.
    For more details, visit: https://fastcrw.com/docs/rest-api

    """

    crw: Any
    api_key: Optional[str]
    api_url: Optional[str]
    mode: Optional[str]
    params: Optional[dict]

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    # --------------------
    # Aux methods (init)
    # --------------------
    def _import_crw(self) -> Any:
        try:
            from crw import CrwClient  # type: ignore
        except Exception as exc:
            raise ImportError(
                "crw not found, please run `pip install crw`"
            ) from exc
        return CrwClient

    def _init_client(self, api_key: Optional[str], api_url: Optional[str]) -> Any:
        CrwClient = self._import_crw()
        client_kwargs: Dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if api_url is not None:
            client_kwargs["api_url"] = api_url
        return CrwClient(**client_kwargs)

    def _params_copy(self) -> Dict[str, Any]:
        params: Dict[str, Any] = self.params.copy() if self.params else {}
        return params

    # --------------------
    # Aux helpers (common)
    # --------------------
    def _safe_get_attr(self, obj: Any, *names: str) -> Optional[Any]:
        for name in names:
            try:
                val = getattr(obj, name, None)
            except Exception:
                val = None
            if val:
                return val
        return None

    def _to_dict_best_effort(self, obj: Any) -> Dict[str, Any]:
        # pydantic v2
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            try:
                return obj.model_dump()  # type: ignore[attr-defined]
            except Exception:
                pass
        # pydantic v1
        if hasattr(obj, "dict") and callable(obj.dict):
            try:
                return obj.dict()  # type: ignore[attr-defined]
            except Exception:
                pass
        # dataclass or simple object
        if hasattr(obj, "__dict__"):
            try:
                return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
            except Exception:
                pass
        # reflect over attributes
        result: Dict[str, Any] = {}
        try:
            for attr in dir(obj):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(obj, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                result[attr] = val
        except Exception:
            pass
        return result

    # --------------------
    # Aux handlers (SCRAPE)
    # --------------------
    def _scrape_get_first(self, data_obj: Dict[str, Any], *keys: str) -> Optional[Any]:
        for k in keys:
            if isinstance(data_obj, dict) and k in data_obj and data_obj.get(k):
                return data_obj.get(k)
        return None

    def _scrape_from_dict(self, crw_docs: Dict[str, Any]) -> (str, Dict[str, Any]):
        data_obj = crw_docs.get("data", crw_docs)
        text_value = (
            self._scrape_get_first(
                data_obj,
                "markdown",
                "content",
                "plainText",
                "html",
                "raw_html",
                "rawHtml",
                "summary",
            )
            or ""
        )

        meta_obj = data_obj.get("metadata", {}) if isinstance(data_obj, dict) else {}
        metadata_value: Dict[str, Any] = {}

        if isinstance(meta_obj, dict):
            metadata_value = meta_obj
        else:
            try:
                metadata_value = self._to_dict_best_effort(meta_obj)
            except Exception:
                metadata_value = {"metadata": str(meta_obj)}

        if isinstance(data_obj, dict):
            for extra_key in (
                "links",
                "json",
                "screenshot",
                "warning",
                "changeTracking",
            ):
                if extra_key in data_obj and data_obj.get(extra_key) is not None:
                    metadata_value[extra_key] = data_obj.get(extra_key)

        if "success" in crw_docs:
            metadata_value["success"] = crw_docs.get("success")
        if "warning" in crw_docs and crw_docs.get("warning") is not None:
            metadata_value["warning_top"] = crw_docs.get("warning")

        return text_value, metadata_value

    def _scrape_from_obj(self, crw_docs: Any) -> (str, Dict[str, Any]):
        text_value = (
            self._safe_get_attr(
                crw_docs,
                "markdown",
                "content",
                "plain_text",
                "html",
                "raw_html",
                "summary",
            )
            or ""
        )

        meta_obj = getattr(crw_docs, "metadata", None)
        metadata_value: Dict[str, Any] = {}
        if meta_obj is not None:
            try:
                metadata_value = self._to_dict_best_effort(meta_obj)
            except Exception:
                metadata_value = {"metadata": str(meta_obj)}

        for extra_attr in (
            "links",
            "json",
            "screenshot",
            "warning",
            "change_tracking",
        ):
            try:
                extra_val = getattr(crw_docs, extra_attr, None)
            except Exception:
                extra_val = None
            if extra_val is not None:
                metadata_value[extra_attr] = extra_val

        return text_value, metadata_value

    def _handle_scrape_response(self, crw_docs: Any) -> (str, Dict[str, Any]):
        if isinstance(crw_docs, dict):
            return self._scrape_from_dict(crw_docs)
        else:
            return self._scrape_from_obj(crw_docs)

    # --------------------
    # __init__ (mirrors firecrawl_web behavior)
    # --------------------
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Optional[str] = "crawl",
        params: Optional[dict] = None,
    ) -> None:
        """Initialize with parameters."""
        # Ensure crw client is installed and instantiate
        try:
            from crw import CrwClient  # type: ignore
        except Exception as exc:
            raise ImportError(
                "crw not found, please run `pip install crw`"
            ) from exc

        # Instantiate the CrwClient. With no api_url it targets the managed
        # cloud (https://fastcrw.com/api) and reads CRW_API_KEY from env.
        client_kwargs: Dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if api_url is not None:
            client_kwargs["api_url"] = api_url

        crw = CrwClient(**client_kwargs)

        params = params or {}
        params["integration"] = "llamaindex"

        super().__init__(
            crw=crw,
            api_key=api_key,
            api_url=api_url,
            mode=mode,
            params=params,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Crw_reader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            url (Optional[str]): URL to scrape or crawl.
            query (Optional[str]): Query to search for.

        Returns:
            List[Document]: List of documents.

        Raises:
            ValueError: If invalid combination of parameters is provided.

        """
        if sum(x is not None for x in [url, query]) != 1:
            raise ValueError("Exactly one of url or query must be provided.")

        documents = []

        if self.mode == "scrape":
            # [SCRAPE] https://fastcrw.com/docs/rest-api (POST /v1/scrape)
            if url is None:
                raise ValueError("URL must be provided for scrape mode.")
            scrape_params = self._params_copy()
            crw_docs = self.crw.scrape(url, **scrape_params)
            text_value, metadata_value = self._handle_scrape_response(crw_docs)
            documents.append(Document(text=text_value or "", metadata=metadata_value))
        elif self.mode == "crawl":
            # [CRAWL] https://fastcrw.com/docs/rest-api (POST /v1/crawl)
            if url is None:
                raise ValueError("URL must be provided for crawl mode.")
            crawl_params = self._params_copy()
            crw_docs = self.crw.crawl(url, **crawl_params)
            # Normalize crawl response: the SDK returns a list of pages, but be
            # defensive about a { data: [...] } envelope as well.
            items: List[Any] = []
            if isinstance(crw_docs, dict):
                data = crw_docs.get("data", crw_docs)
                if isinstance(data, list):
                    items = data
            elif isinstance(crw_docs, list):
                items = crw_docs
            else:
                for attr_name in ("data", "results", "documents", "items", "pages"):
                    try:
                        candidate = getattr(crw_docs, attr_name, None)
                    except Exception:
                        candidate = None
                    if isinstance(candidate, list) and candidate:
                        items = candidate
                        break

            for doc in items:
                if isinstance(doc, dict):
                    text_val = (
                        doc.get("markdown")
                        or doc.get("content")
                        or doc.get("text")
                        or ""
                    )
                    metadata_val = doc.get("metadata", {})
                else:
                    text_val = (
                        self._safe_get_attr(
                            doc,
                            "markdown",
                            "content",
                            "text",
                            "html",
                            "raw_html",
                            "rawHtml",
                            "summary",
                        )
                        or ""
                    )
                    meta_obj = getattr(doc, "metadata", None)
                    if isinstance(meta_obj, dict):
                        metadata_val = meta_obj
                    elif meta_obj is not None:
                        try:
                            metadata_val = self._to_dict_best_effort(meta_obj)
                        except Exception:
                            metadata_val = {"metadata": str(meta_obj)}
                    else:
                        metadata_val = {}
                documents.append(Document(text=text_val, metadata=metadata_val))
        elif self.mode == "map":
            # [MAP] https://fastcrw.com/docs/rest-api (POST /v1/map)
            # The SDK returns a list of link strings; be defensive about a
            # { success, links: [...] } envelope or an object with `links`.
            if url is None:
                raise ValueError("URL must be provided for map mode.")

            map_params = self._params_copy()
            response = self.crw.map(url, **map_params)  # type: ignore[attr-defined]

            # Handle error response format: { "error": "..." }
            if (
                isinstance(response, dict)
                and "error" in response
                and not response.get("success", False)
            ):
                error_message = response.get("error", "Unknown error")
                documents.append(
                    Document(
                        text=f"Map request failed: {error_message}",
                        metadata={"source": "map", "url": url, "error": error_message},
                    )
                )
                return documents

            # Extract links across the supported response shapes
            if isinstance(response, list):
                links = response
            elif isinstance(response, dict):
                links = response.get("links", []) or []
            else:
                links = getattr(response, "links", None) or []

            for link in links:
                if isinstance(link, str):
                    link_url = link
                    title = ""
                    description = ""
                elif isinstance(link, dict):
                    link_url = link.get("url", "")
                    title = link.get("title", "")
                    description = link.get("description", "")
                else:
                    link_url = getattr(link, "url", "")
                    title = getattr(link, "title", "")
                    description = getattr(link, "description", "")
                text_content = title or description or link_url
                documents.append(
                    Document(
                        text=text_content,
                        metadata={
                            "source": "map",
                            "url": link_url,
                            "title": title,
                            "description": description,
                        },
                    )
                )
        elif self.mode == "search":
            # [SEARCH] https://fastcrw.com/docs/rest-api (POST /v1/search)
            if query is None:
                raise ValueError("Query must be provided for search mode.")

            # Remove query from params if it exists to avoid duplicate
            search_params = self._params_copy()
            if "query" in search_params:
                del search_params["query"]

            search_response = self.crw.search(query, **search_params)

            # The SDK returns a list of result dicts; be defensive about a
            # { success, data: [...] } envelope as well.
            if isinstance(search_response, list):
                search_results = search_response
            elif isinstance(search_response, dict):
                if not search_response.get("success", True):
                    warning = search_response.get("warning", "Unknown error")
                    print(f"Search was unsuccessful: {warning}")
                    documents.append(
                        Document(
                            text=f"Search for '{query}' was unsuccessful: {warning}",
                            metadata={
                                "source": "search",
                                "query": query,
                                "error": warning,
                            },
                        )
                    )
                    return documents
                search_results = search_response.get("data", [])
            else:
                print(f"Unexpected search response format: {type(search_response)}")
                documents.append(
                    Document(
                        text=str(search_response),
                        metadata={"source": "search", "query": query},
                    )
                )
                return documents

            for result in search_results:
                if isinstance(result, dict):
                    text = result.get("markdown", "")
                    if not text:
                        text = result.get("description", "")
                    metadata = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "source": "search",
                        "query": query,
                    }
                    if "metadata" in result and isinstance(result["metadata"], dict):
                        metadata.update(result["metadata"])
                else:
                    item_url = getattr(result, "url", "")
                    item_title = getattr(result, "title", "")
                    item_description = getattr(result, "description", "")
                    text = (
                        self._safe_get_attr(result, "markdown")
                        or item_description
                        or ""
                    )
                    metadata = {
                        "title": item_title,
                        "url": item_url,
                        "description": item_description,
                        "source": "search",
                        "query": query,
                    }
                    base_keys = set(metadata.keys())
                    extra_attrs = self._to_dict_best_effort(result)
                    for k, v in extra_attrs.items():
                        if k not in base_keys:
                            metadata[k] = v
                documents.append(Document(text=text, metadata=metadata))
        else:
            raise ValueError(
                "Invalid mode. Please choose 'scrape', 'crawl', 'map', or 'search'."
            )

        return documents
