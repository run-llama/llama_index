"""An asynchronous Solr client implementation using ``aiosolr`` under the hood."""

import asyncio
import logging
import sys
import time
from asyncio import Task
from collections.abc import Mapping, Sequence
from typing import Any, Union, cast
from urllib.parse import urlparse

import aiosolr
from pydantic import ValidationError

from llama_index.vector_stores.solr.client._base import _BaseSolrClient
from llama_index.vector_stores.solr.client.responses import (
    SolrSelectResponse,
    SolrUpdateResponse,
)
from llama_index.vector_stores.solr.client.utils import prepare_document_for_solr
from llama_index.vector_stores.solr.constants import SolrConstants

logger = logging.getLogger(__name__)


class AsyncSolrClient(_BaseSolrClient):
    """
    A Solr client that wraps :py:class:`aiosolr.Client`.

    See `aiosolr <https://github.com/youversion/aiosolr>`_ for implementation details.
    """

    async def _build_client(self) -> aiosolr.Client:
        try:
            logger.info("Initializing aiosolr client for URL: %s", self.base_url)
            # aiosolr.Client builds URLs for various actions in a hardcoded manner; for
            # URLs with ports (such as localhost URLs), we need to pass the parsed version
            # for external URLs, we need to pass the connection URL directly
            parsed_url = urlparse(self.base_url)
            *_, collection = parsed_url.path.split("/")
            if parsed_url.port is not None:
                args = {
                    "host": parsed_url.hostname,
                    "port": parsed_url.port,
                    "scheme": parsed_url.scheme,
                    "collection": collection,
                    **self._client_kwargs,
                }
            else:
                args = {
                    "connection_url": self._base_url,
                    **self._client_kwargs,
                }

            if sys.version_info < (3, 10):
                args["timeout"] = self._request_timeout_sec
            else:
                args["read_timeout"] = self._request_timeout_sec
                args["write_timeout"] = self._request_timeout_sec

            logger.debug("Initializing AIOSolr client with args: %s", self._base_url)
            client = aiosolr.Client(**args)
            await client.setup()
            # should not happen
            if client.session is None:  # pragma: no cover
                raise ValueError("AIOSolr client session was not created after setup")

            if self._headers:
                client.session.headers.update(self._headers)
                logger.debug(
                    "Updated AIOSolr client default headers with keys: %s",
                    list(self._headers.keys()),
                )
            return client

        except RuntimeError as exc:  # pragma: no cover
            raise ValueError(
                f"AIOSolr client cannot be initialized (likely due to running in "
                f"non-async context), type={type(exc)} err={exc}"
            ) from exc

    async def _get_client(self) -> aiosolr.Client:
        # defer session creation until actually required
        if not self._client:
            self._client = await self._build_client()
        return self._client

    async def search(
        self, query_params: Mapping[str, Any], **kwargs: Any
    ) -> SolrSelectResponse:
        """
        Asynchronously search Solr with the input query, returning any matching documents.

        No validation is done on the input query dictionary.

        Args:
            query_params: A query dictionary to be sent to Solr.
            **kwargs:
                Additional keyword arguments to pass to :py:meth:`aiosolr.Client.query`.

        Returns:
            The deserialized response from Solr.

        """
        try:
            logger.info("Searching Solr with query='%s'", query_params)
            client = await self._get_client()
            results = await client.query(**query_params, **kwargs)
            response = SolrSelectResponse.from_aiosolr_response(results)
            logger.info(
                "Solr response received (path=select): status=%s qtime=%s hits=%s",
                response.response_header.status,
                response.response_header.q_time,
                response.response.num_found,
            )
            return response
        except aiosolr.SolrError as err:
            raise ValueError(
                f"Error during Aiosolr call, type={type(err)} err={err}"
            ) from err
        except ValidationError as err:
            raise ValueError(
                f"Unexpected response format from Solr: err={err.json()}"
            ) from err

    async def add(
        self, documents: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> SolrUpdateResponse:
        """
        Asynchronously add documents to the Solr collection.

        No validation is done on the input documents.

        Args:
            documents:
                The documents to be added to the Solr collection. These documents should
                be serializable to JSON.
            **kwargs:
                Additional keyword arguments to be passed to :py:meth:`aiosolr.Client.add`.

        Returns:
            The deserialized update response from Solr.

        """
        logger.debug("Preparing documents for insertion into Solr collection")
        start = time.perf_counter()
        updated_docs = [prepare_document_for_solr(doc) for doc in documents]
        logger.debug(
            "Prepared %d documents, took %.2g seconds",
            len(documents),
            time.perf_counter() - start,
        )

        try:
            logger.info("Adding %d documents to the Solr collection", len(documents))
            client = await self._get_client()
            results = await client.update(data=updated_docs, **kwargs)
            response = SolrUpdateResponse.from_aiosolr_response(results)
            logger.info(
                "Solr response received (path=update): status=%s",
                response.response_header.status,
            )
            return response
        except aiosolr.SolrError as err:
            raise ValueError(
                f"Error during Aiosolr call, type={type(err)} err={err}"
            ) from err
        except ValidationError as err:
            raise ValueError(
                f"Unexpected response format from Solr: err={err.json()}"
            ) from err

    async def _delete(
        self, delete_command: Union[list[str], dict[str, Any]], **kwargs: Any
    ) -> SolrUpdateResponse:
        try:
            client = await self._get_client()
            delete_query = {"delete": delete_command}
            results = await client.update(data=delete_query, **kwargs)
            response = SolrUpdateResponse.from_aiosolr_response(results)
            logger.info(
                "Solr response received (path=update): status=%s qtime=%s",
                response.response_header.status,
                response.response_header.q_time,
            )
            return response
        except aiosolr.SolrError as err:
            raise ValueError(
                f"Error during Aiosolr call, type={type(err)} err={err}"
            ) from err
        except ValidationError as err:
            raise ValueError(
                f"Unexpected response format from Solr: err={err.json()}"
            ) from err

    async def delete_by_query(
        self, query_string: str, **kwargs: Any
    ) -> SolrUpdateResponse:
        """
        Asynchronously delete documents from the Solr collection using a query string.

        No validation is done on the input query string.

        Args:
            query_string: A query string matching the documents that should be deleted.
            **kwargs:
                Additional keyword arguments to be passed to
                :py:meth:`aiosolr.Client.update`.

        Returns:
            The deserialized response from Solr.

        """
        logger.info(
            "Deleting documents from Solr matching query '%s', collection url=%s",
            query_string,
            self._base_url,
        )
        return await self._delete({"query": query_string}, **kwargs)

    async def delete_by_id(
        self, ids: Sequence[str], **kwargs: Any
    ) -> SolrUpdateResponse:
        """
        Asynchronously delete documents from the Solr collection using their IDs.

        If the set of IDs is known, this is generally more efficient than using
        :py:meth:`.delete_by_query`.

        Args:
            ids: A sequence of document IDs to be deleted.
            **kwargs:
                Additional keyword arguments to be passed to
                :py:meth:`aiosolr.Client.update`.

        Returns:
            The deserialized response from Solr.

        Raises:
            ValueError: If the list of IDs is empty.

        """
        if not ids:
            raise ValueError("The list of IDs to delete cannot be empty")

        logger.info(
            "Deleting %d documents from the Solr collection by ID, collection url=%s",
            len(ids),
            self._base_url,
        )
        return await self._delete(list(ids), **kwargs)

    async def clear_collection(self, **kwargs) -> SolrUpdateResponse:
        """
        Asynchronously delete all documents from the Solr collection.

        Args:
            **kwargs:
                Optional keyword arguments to be passed to
                :py:meth:`aiosolr.Client.update`.

        Returns:
            The deserialized response from Solr.

        """
        return await self.delete_by_query(SolrConstants.QUERY_ALL, **kwargs)

    async def close(self) -> None:
        """Close the ``aiosolr`` client, if it exists."""
        if self._client is not None:
            await cast(aiosolr.Client, self._client).close()

    def __del__(self) -> None:
        """Destroy the client, ensuring the session gets closed if it's not already."""
        tasks: set[Task] = set()
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                task = loop.create_task(self.close())
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            else:  # pragma: no cover
                loop.run_until_complete(self.close())
                return
        except RuntimeError as exc:
            logger.debug(
                "No running event loop, nothing to close, type=%s err='%s'",
                type(exc),
                exc,
            )
        # last resort catch for interpreter shutdown, not reasonably testable
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to close the async Solr client, type=%s err='%s'",
                type(exc),
                exc,
            )
