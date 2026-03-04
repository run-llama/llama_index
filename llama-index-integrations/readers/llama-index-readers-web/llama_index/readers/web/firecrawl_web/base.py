"""Firecrawl Web Reader."""

from typing import Any, List, Optional, Dict, Callable

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class FireCrawlWebReader(BasePydanticReader):
    """
    turn a url to llm accessible markdown with `Firecrawl.dev`.

    Args:
        api_key (str): The Firecrawl API key.
        api_url (Optional[str]): Optional base URL for Firecrawl deployment
        mode (Optional[str]):
            The mode to run the loader in. Default is "crawl".
            Options include "scrape" (single url),
            "crawl" (all accessible sub pages),
            "map" (map all accessible sub pages),
            "search" (search for content), and
            "extract" (extract structured data from URLs using a prompt).
        params (Optional[dict]): The parameters to pass to the Firecrawl API.

    Examples include crawlerOptions.
    For more details, visit: https://docs.firecrawl.dev/sdks/python

    """

    firecrawl: Any
    api_key: str
    api_url: Optional[str]
    mode: Optional[str]
    params: Optional[dict]

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    # --------------------
    # Aux methods (init)
    # --------------------
    def _import_firecrawl(self) -> Any:
        try:
            from firecrawl import Firecrawl  # type: ignore
        except Exception as exc:
            raise ImportError(
                "firecrawl not found, please run `pip install 'firecrawl-py>=4.3.3'`"
            ) from exc
        return Firecrawl

    def _init_client(self, api_key: str, api_url: Optional[str]) -> Any:
        Firecrawl = self._import_firecrawl()
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_url is not None:
            client_kwargs["api_url"] = api_url
        return Firecrawl(**client_kwargs)

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

    def _scrape_from_dict(
        self, firecrawl_docs: Dict[str, Any]
    ) -> (str, Dict[str, Any]):
        data_obj = firecrawl_docs.get("data", firecrawl_docs)
        text_value = (
            self._scrape_get_first(
                data_obj,
                "markdown",
                "content",
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
                "actions",
                "screenshot",
                "warning",
                "changeTracking",
            ):
                if extra_key in data_obj and data_obj.get(extra_key) is not None:
                    metadata_value[extra_key] = data_obj.get(extra_key)

        if "success" in firecrawl_docs:
            metadata_value["success"] = firecrawl_docs.get("success")
        if "warning" in firecrawl_docs and firecrawl_docs.get("warning") is not None:
            metadata_value["warning_top"] = firecrawl_docs.get("warning")

        return text_value, metadata_value

    def _scrape_from_obj(self, firecrawl_docs: Any) -> (str, Dict[str, Any]):
        text_value = (
            self._safe_get_attr(
                firecrawl_docs,
                "markdown",
                "content",
                "html",
                "raw_html",
                "summary",
            )
            or ""
        )

        meta_obj = getattr(firecrawl_docs, "metadata", None)
        metadata_value: Dict[str, Any] = {}
        if meta_obj is not None:
            try:
                metadata_value = self._to_dict_best_effort(meta_obj)
            except Exception:
                metadata_value = {"metadata": str(meta_obj)}

        for extra_attr in (
            "links",
            "actions",
            "screenshot",
            "warning",
            "change_tracking",
        ):
            try:
                extra_val = getattr(firecrawl_docs, extra_attr, None)
            except Exception:
                extra_val = None
            if extra_val is not None:
                metadata_value[extra_attr] = extra_val

        return text_value, metadata_value

    def _handle_scrape_response(self, firecrawl_docs: Any) -> (str, Dict[str, Any]):
        if isinstance(firecrawl_docs, dict):
            return self._scrape_from_dict(firecrawl_docs)
        else:
            return self._scrape_from_obj(firecrawl_docs)

    # --------------------
    # Aux handlers (CRAWL)
    # --------------------
    def _normalize_crawl_response(self, firecrawl_docs: Any) -> List[Dict[str, Any]]:
        return firecrawl_docs.get("data", firecrawl_docs)

    # --------------------
    # Aux handlers (MAP)
    # --------------------
    def _handle_map_error_or_links(self, response: Any, url: str) -> List[Document]:
        docs: List[Document] = []
        if (
            isinstance(response, dict)
            and "error" in response
            and not response.get("success", False)
        ):
            error_message = response.get("error", "Unknown error")
            docs.append(
                Document(
                    text=f"Map request failed: {error_message}",
                    metadata={"source": "map", "url": url, "error": error_message},
                )
            )
            return docs

        links = response.links or []
        for link in links:
            link_url = link.url
            title = link.title
            description = link.description
            text_content = title or description or link_url
            docs.append(
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
        return docs

    # --------------------
    # Aux handlers (SEARCH)
    # --------------------
    def _process_search_dict(
        self, search_response: Dict[str, Any], query: str
    ) -> List[Document]:
        documents: List[Document] = []
        if search_response.get("success", False):
            search_results = search_response.get("data", [])
            for result in search_results:
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
                documents.append(Document(text=text, metadata=metadata))
        else:
            warning = search_response.get("warning", "Unknown error")
            print(f"Search was unsuccessful: {warning}")
            documents.append(
                Document(
                    text=f"Search for '{query}' was unsuccessful: {warning}",
                    metadata={"source": "search", "query": query, "error": warning},
                )
            )
        return documents

    def _process_search_items(
        self, result_list: Any, result_type: str, query: str
    ) -> List[Document]:
        docs: List[Document] = []
        if not result_list:
            return docs
        for item in result_list:
            item_url = getattr(item, "url", "")
            item_title = getattr(item, "title", "")
            item_description = getattr(item, "description", "")
            text_content = item_title or item_description or item_url

            metadata = {
                "title": item_title,
                "url": item_url,
                "description": item_description,
                "source": "search",
                "search_type": result_type,
                "query": query,
            }
            base_keys = set(metadata.keys())
            extra_attrs = self._to_dict_best_effort(item)
            for k, v in extra_attrs.items():
                if k not in base_keys:
                    metadata[k] = v
            docs.append(Document(text=text_content, metadata=metadata))
        return docs

    def _process_search_sdk(self, search_response: Any, query: str) -> List[Document]:
        documents: List[Document] = []
        documents += self._process_search_items(
            getattr(search_response, "web", None), "web", query
        )  # type: ignore[attr-defined]
        documents += self._process_search_items(
            getattr(search_response, "news", None), "news", query
        )  # type: ignore[attr-defined]
        documents += self._process_search_items(
            getattr(search_response, "images", None), "images", query
        )  # type: ignore[attr-defined]
        return documents

    # --------------------
    # Aux handlers (EXTRACT)
    # --------------------
    def _format_extract_text(self, extract_data: Dict[str, Any]) -> str:
        text_parts = []
        for key, value in extract_data.items():
            text_parts.append(f"{key}: {value}")
        return "\n".join(text_parts)

    # --------------------
    # __init__ (unchanged behavior)
    # --------------------
    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        mode: Optional[str] = "crawl",
        params: Optional[dict] = None,
    ) -> None:
        """Initialize with parameters."""
        # Ensure firecrawl client is installed and instantiate
        try:
            from firecrawl import Firecrawl  # type: ignore
        except Exception as exc:
            raise ImportError(
                "firecrawl not found, please run `pip install 'firecrawl-py>=4.3.3'`"
            ) from exc

        # Instantiate the new Firecrawl client
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_url is not None:
            client_kwargs["api_url"] = api_url

        firecrawl = Firecrawl(**client_kwargs)

        params = params or {}
        params["integration"] = "llamaindex"

        super().__init__(
            firecrawl=firecrawl,
            api_key=api_key,
            api_url=api_url,
            mode=mode,
            params=params,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Firecrawl_reader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        urls: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            url (Optional[str]): URL to scrape or crawl.
            query (Optional[str]): Query to search for.
            urls (Optional[List[str]]): List of URLs for extract mode.

        Returns:
            List[Document]: List of documents.

        Raises:
            ValueError: If invalid combination of parameters is provided.

        """
        if sum(x is not None for x in [url, query, urls]) != 1:
            raise ValueError("Exactly one of url, query, or urls must be provided.")

        documents = []

        if self.mode == "scrape":
            # [SCRAPE] params: https://docs.firecrawl.dev/api-reference/endpoint/scrape
            if url is None:
                raise ValueError("URL must be provided for scrape mode.")
            # Map params to new client call signature
            scrape_params = self._params_copy()
            firecrawl_docs = self.firecrawl.scrape(url, **scrape_params)
            # Support both dict and SDK object responses
            text_value = ""
            metadata_value: Dict[str, Any] = {}

            if isinstance(firecrawl_docs, dict):
                # Newer API may return { success, data: {...} }
                data_obj = firecrawl_docs.get("data", firecrawl_docs)

                def _get_first(*keys: str) -> Optional[Any]:
                    for k in keys:
                        if (
                            isinstance(data_obj, dict)
                            and k in data_obj
                            and data_obj.get(k)
                        ):
                            return data_obj.get(k)
                    return None

                text_value = (
                    _get_first(
                        "markdown", "content", "html", "raw_html", "rawHtml", "summary"
                    )
                    or ""
                )

                meta_obj = (
                    data_obj.get("metadata", {}) if isinstance(data_obj, dict) else {}
                )
                if isinstance(meta_obj, dict):
                    metadata_value = meta_obj
                else:
                    # Convert metadata object to dict if needed
                    try:
                        if hasattr(meta_obj, "model_dump") and callable(
                            meta_obj.model_dump
                        ):
                            metadata_value = meta_obj.model_dump()  # type: ignore[attr-defined]
                        elif hasattr(meta_obj, "dict") and callable(meta_obj.dict):
                            metadata_value = meta_obj.dict()  # type: ignore[attr-defined]
                        elif hasattr(meta_obj, "__dict__"):
                            metadata_value = {
                                k: v
                                for k, v in vars(meta_obj).items()
                                if not k.startswith("_")
                            }
                    except Exception:
                        metadata_value = {"metadata": str(meta_obj)}

                # Capture other helpful fields into metadata
                if isinstance(data_obj, dict):
                    for extra_key in (
                        "links",
                        "actions",
                        "screenshot",
                        "warning",
                        "changeTracking",
                    ):
                        if (
                            extra_key in data_obj
                            and data_obj.get(extra_key) is not None
                        ):
                            metadata_value[extra_key] = data_obj.get(extra_key)
                # Bubble up success/warning if at top-level
                if "success" in firecrawl_docs:
                    metadata_value["success"] = firecrawl_docs.get("success")
                if (
                    "warning" in firecrawl_docs
                    and firecrawl_docs.get("warning") is not None
                ):
                    metadata_value["warning_top"] = firecrawl_docs.get("warning")
            else:
                # SDK object with attributes
                def _safe_get(obj: Any, *names: str) -> Optional[Any]:
                    for name in names:
                        try:
                            val = getattr(obj, name, None)
                        except Exception:
                            val = None
                        if val:
                            return val
                    return None

                text_value = (
                    _safe_get(
                        firecrawl_docs,
                        "markdown",
                        "content",
                        "html",
                        "raw_html",
                        "summary",
                    )
                    or ""
                )

                meta_obj = getattr(firecrawl_docs, "metadata", None)
                if meta_obj is not None:
                    try:
                        if hasattr(meta_obj, "model_dump") and callable(
                            meta_obj.model_dump
                        ):
                            metadata_value = meta_obj.model_dump()  # type: ignore[attr-defined]
                        elif hasattr(meta_obj, "dict") and callable(meta_obj.dict):
                            metadata_value = meta_obj.dict()  # type: ignore[attr-defined]
                        elif hasattr(meta_obj, "__dict__"):
                            metadata_value = {
                                k: v
                                for k, v in vars(meta_obj).items()
                                if not k.startswith("_")
                            }
                        else:
                            metadata_value = {"metadata": str(meta_obj)}
                    except Exception:
                        metadata_value = {"metadata": str(meta_obj)}

                # Attach extra top-level attributes if present on SDK object
                for extra_attr in (
                    "links",
                    "actions",
                    "screenshot",
                    "warning",
                    "change_tracking",
                ):
                    try:
                        extra_val = getattr(firecrawl_docs, extra_attr, None)
                    except Exception:
                        extra_val = None
                    if extra_val is not None:
                        metadata_value[extra_attr] = extra_val

            documents.append(Document(text=text_value or "", metadata=metadata_value))
        elif self.mode == "crawl":
            # [CRAWL] params: https://docs.firecrawl.dev/api-reference/endpoint/crawl-post
            if url is None:
                raise ValueError("URL must be provided for crawl mode.")
            crawl_params = self._params_copy()
            # Remove deprecated/unsupported parameters
            if "maxDepth" in crawl_params:
                crawl_params.pop("maxDepth", None)
            firecrawl_docs = self.firecrawl.crawl(url, **crawl_params)
            # Normalize Crawl response across SDK versions
            items: List[Any] = []
            if isinstance(firecrawl_docs, dict):
                data = firecrawl_docs.get("data", firecrawl_docs)
                if isinstance(data, list):
                    items = data
            else:
                # Try common list-bearing attributes first
                for attr_name in ("data", "results", "documents", "items", "pages"):
                    try:
                        candidate = getattr(firecrawl_docs, attr_name, None)
                    except Exception:
                        candidate = None
                    if isinstance(candidate, list) and candidate:
                        items = candidate
                        break
                # Fallback to model dump reflection
                if not items:
                    try:
                        if hasattr(firecrawl_docs, "model_dump") and callable(
                            firecrawl_docs.model_dump
                        ):
                            dump_obj = firecrawl_docs.model_dump()  # type: ignore[attr-defined]
                        elif hasattr(firecrawl_docs, "dict") and callable(
                            firecrawl_docs.dict
                        ):
                            dump_obj = firecrawl_docs.dict()  # type: ignore[attr-defined]
                        else:
                            dump_obj = {}
                    except Exception:
                        dump_obj = {}
                    if isinstance(dump_obj, dict):
                        data = (
                            dump_obj.get("data")
                            or dump_obj.get("results")
                            or dump_obj.get("documents")
                        )
                        if isinstance(data, list):
                            items = data

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
            # [MAP] params: https://docs.firecrawl.dev/api-reference/endpoint/map
            # Expected response: { "success": true, "links": [{"url":..., "title":..., "description":...}, ...] }
            if url is None:
                raise ValueError("URL must be provided for map mode.")

            map_params = self._params_copy()
            # Pass through optional parameters like sitemap, includeSubdomains, ignoreQueryParameters, limit, timeout, search
            response = self.firecrawl.map(url, **map_params)  # type: ignore[attr-defined]

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

            # Extract links from success response
            links = response.links or []

            for link in links:
                link_url = link.url
                title = link.title
                description = link.description
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
            # [SEARCH] params: https://docs.firecrawl.dev/api-reference/endpoint/search
            if query is None:
                raise ValueError("Query must be provided for search mode.")

            # Remove query from params if it exists to avoid duplicate
            search_params = self._params_copy()
            if "query" in search_params:
                del search_params["query"]

            # Get search results
            search_response = self.firecrawl.search(query, **search_params)

            # Handle the search response format
            if isinstance(search_response, dict):
                # Check for success
                if search_response.get("success", False):
                    # Get the data array
                    search_results = search_response.get("data", [])

                    # Process each search result
                    for result in search_results:
                        # Extract text content (prefer markdown if available)
                        text = result.get("markdown", "")
                        if not text:
                            # Fall back to description if markdown is not available
                            text = result.get("description", "")

                        # Extract metadata
                        metadata = {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", ""),
                            "source": "search",
                            "query": query,
                        }

                        # Add additional metadata if available
                        if "metadata" in result and isinstance(
                            result["metadata"], dict
                        ):
                            metadata.update(result["metadata"])

                        # Create document
                        documents.append(
                            Document(
                                text=text,
                                metadata=metadata,
                            )
                        )
                else:
                    # Handle unsuccessful response
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
            elif (
                hasattr(search_response, "web")
                or hasattr(search_response, "news")
                or hasattr(search_response, "images")
            ):
                # New SDK object response like: web=[SearchResultWeb(...)] news=None images=None
                def _process_results(result_list, result_type: str) -> None:
                    if not result_list:
                        return
                    for item in result_list:
                        # Try to access attributes with safe fallbacks
                        item_url = getattr(item, "url", "")
                        item_title = getattr(item, "title", "")
                        item_description = getattr(item, "description", "")
                        text_content = item_title or item_description or item_url

                        metadata = {
                            "title": item_title,
                            "url": item_url,
                            "description": item_description,
                            "source": "search",
                            "search_type": result_type,
                            "query": query,
                        }

                        # Collect all other attributes dynamically without whitelisting
                        base_keys = set(metadata.keys())

                        def _item_to_dict(obj: Any) -> Dict[str, Any]:
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
                                    return {
                                        k: v
                                        for k, v in vars(obj).items()
                                        if not k.startswith("_")
                                    }
                                except Exception:
                                    pass
                            # Fallback: reflect over attributes
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

                        extra_attrs = _item_to_dict(item)
                        for k, v in extra_attrs.items():
                            if k not in base_keys:
                                metadata[k] = v

                        documents.append(
                            Document(
                                text=text_content,
                                metadata=metadata,
                            )
                        )

                _process_results(getattr(search_response, "web", None), "web")  # type: ignore[attr-defined]
                _process_results(getattr(search_response, "news", None), "news")  # type: ignore[attr-defined]
                _process_results(getattr(search_response, "images", None), "images")  # type: ignore[attr-defined]
            else:
                # Handle unexpected response format
                print(f"Unexpected search response format: {type(search_response)}")
                documents.append(
                    Document(
                        text=str(search_response),
                        metadata={"source": "search", "query": query},
                    )
                )
        elif self.mode == "extract":
            # [EXTRACT] params: https://docs.firecrawl.dev/api-reference/endpoint/extract
            if urls is None:
                # For backward compatibility, convert single URL to list if provided
                if url is not None:
                    urls = [url]
                else:
                    raise ValueError("URLs must be provided for extract mode.")

            # Ensure we have a prompt in params
            extract_params = self._params_copy()
            if "prompt" not in extract_params:
                raise ValueError("A 'prompt' parameter is required for extract mode.")

            # Prepare the payload according to the new API structure
            payload = {"prompt": extract_params.pop("prompt")}
            payload["integration"] = "llamaindex"

            # Call the extract method with the urls and params
            extract_response = self.firecrawl.extract(urls=urls, **payload)

            # Handle the extract response format
            if isinstance(extract_response, dict):
                # Check for success
                if extract_response.get("success", False):
                    # Get the data from the response
                    extract_data = extract_response.get("data", {})

                    # Get the sources if available
                    sources = extract_response.get("sources", {})

                    # Convert the extracted data to text
                    if extract_data:
                        # Convert the data to a formatted string
                        text_parts = []
                        for key, value in extract_data.items():
                            text_parts.append(f"{key}: {value}")

                        text = "\n".join(text_parts)

                        # Create metadata
                        metadata = {
                            "urls": urls,
                            "source": "extract",
                            "status": extract_response.get("status"),
                            "expires_at": extract_response.get("expiresAt"),
                        }

                        # Add sources to metadata if available
                        if sources:
                            metadata["sources"] = sources

                        # Create document
                        documents.append(
                            Document(
                                text=text,
                                metadata=metadata,
                            )
                        )
                    else:
                        # Handle empty data in successful response
                        print("Extract response successful but no data returned")
                        documents.append(
                            Document(
                                text="Extraction was successful but no data was returned",
                                metadata={"urls": urls, "source": "extract"},
                            )
                        )
                else:
                    # Handle unsuccessful response
                    warning = extract_response.get("warning", "Unknown error")
                    print(f"Extraction was unsuccessful: {warning}")
                    documents.append(
                        Document(
                            text=f"Extraction was unsuccessful: {warning}",
                            metadata={
                                "urls": urls,
                                "source": "extract",
                                "error": warning,
                            },
                        )
                    )
            else:
                # Handle unexpected response format
                print(f"Unexpected extract response format: {type(extract_response)}")
                documents.append(
                    Document(
                        text=str(extract_response),
                        metadata={"urls": urls, "source": "extract"},
                    )
                )
        else:
            raise ValueError(
                "Invalid mode. Please choose 'scrape', 'crawl', 'search', or 'extract'."
            )

        return documents
