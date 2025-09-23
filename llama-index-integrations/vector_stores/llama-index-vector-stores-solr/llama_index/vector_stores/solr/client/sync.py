"""A synchronous Solr client implementation using ``pysolr`` under the hood."""

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from xml.etree.ElementTree import ParseError

import pysolr
from pydantic import ValidationError

from llama_index.vector_stores.solr.client._base import _BaseSolrClient
from llama_index.vector_stores.solr.client.responses import (
    SolrSelectResponse,
    SolrUpdateResponse,
)
from llama_index.vector_stores.solr.client.utils import prepare_document_for_solr
from llama_index.vector_stores.solr.constants import SolrConstants

logger = logging.getLogger(__name__)


class SyncSolrClient(_BaseSolrClient):
    """
    A synchronous Solr client that wraps :py:class:`pysolr.Solr`.

    See `pysolr <https://github.com/django-haystack/pysolr/blob/master/pysolr.py>`_ for
    implementation details.
    """

    def _get_client(self) -> pysolr.Solr:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> pysolr.Solr:
        logger.info("Initializing pysolr client for URL: %s", self.base_url)
        client = pysolr.Solr(
            url=self.base_url, timeout=self._request_timeout_sec, **self._client_kwargs
        )
        if self._headers:
            session = client.get_session()
            session.headers.update(self._headers)
            logger.debug(
                "Updated pysolr client default headers with keys: %s",
                list(self._headers.keys()),
            )
        return client

    def close(self) -> None:
        """Close the underlying Solr client session."""
        if self._client:
            logger.debug("Closing the Solr client session")
            # pysolr doesn't expose a close method, so we directly close the underlying session
            self._client.get_session().close()
            self._client = None

    def search(
        self, query_params: Mapping[str, Any], **kwargs: Any
    ) -> SolrSelectResponse:
        """
        Search Solr with the input query, returning any matching documents.

        No validation is done on the input query dictionary.

        Args:
            query_params: A query dictionary to be sent to Solr.
            **kwargs:
                Additional keyword arguments to pass to :py:meth:`pysolr.Solr.search`.

        Returns:
            The deserialized response from Solr.

        """
        try:
            logger.info("Searching Solr with query='%s'", query_params)
            results = self._get_client().search(**query_params, **kwargs)
            response = SolrSelectResponse.from_pysolr_results(results)
            logger.info(
                "Solr response received (path=select): status=%s qtime=%s hits=%s",
                response.response_header.status,
                response.response_header.q_time,
                response.response.num_found,
            )
            return response
        except pysolr.SolrError as err:
            raise ValueError(
                f"Error during Pysolr call, type={type(err)} err={err}"
            ) from err
        except ValidationError as err:
            raise ValueError(
                f"Unexpected response format from Solr: err={err.json()}"
            ) from err

    def add(
        self, documents: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> SolrUpdateResponse:
        """
        Add documents to the Solr collection.

        No validation is done on the input documents.

        Args:
            documents:
                The documents to be added to the Solr collection. These documents should
                be serializable to JSON.
            **kwargs:
                Additional keyword arguments to pass to :py:meth:`pysolr.Solr.add`.

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
            # pysolr.Solr.add is not typed, but in code tracing it will always be this
            res_text = str(self._get_client().add(updated_docs, **kwargs))
            # update responses in pysolr are always in XML format
            # response = SolrUpdateResponse.from_xml(res_text)
            response = SolrUpdateResponse.model_validate_json(res_text)
            logger.info(
                "Solr response received (path=update): status=%s qtime=%s",
                response.response_header.status,
                response.response_header.q_time,
            )
            return response
        except pysolr.SolrError as err:
            raise ValueError(
                f"Error during Pysolr call, type={type(err)} err={err}"
            ) from err
        except ValidationError as err:
            raise ValueError(
                f"Unexpected response format from Solr: err={err.json()}"
            ) from err

    def _delete(
        self, query_string: Optional[str], ids: Optional[list[str]], **kwargs: Any
    ) -> SolrUpdateResponse:
        try:
            res_text = self._get_client().delete(q=query_string, id=ids, **kwargs)
            # update responses in pysolr are always in XML format
            response = SolrUpdateResponse.from_xml(res_text)
            logger.info(
                "Solr response received (path=update): status=%s qtime=%s",
                response.response_header.status,
                response.response_header.q_time,
            )
            return response
        except pysolr.SolrError as err:
            raise ValueError(
                f"Error during Pysolr call, type={type(err)} err={err}"
            ) from err
        except ParseError as err:
            raise ValueError(
                f"Error parsing XML response from Solr: err={err}"
            ) from err
        except ValidationError as err:
            raise ValueError(
                f"Unexpected response format from Solr: err={err.json()}"
            ) from err

    def delete_by_query(self, query_string: str, **kwargs: Any) -> SolrUpdateResponse:
        """
        Delete documents from the Solr collection using a query string.

        Args:
            query_string: A query string matching the documents that should be deleted.
            **kwargs:
                Additional keyword arguments to pass to :py:meth:`pysolr.Solr.delete`.

        Returns:
            The deserialized response from Solr.

        """
        logger.info(
            "Deleting documents from Solr matching query '%s', collection url=%s",
            query_string,
            self._base_url,
        )
        return self._delete(query_string=query_string, ids=None, **kwargs)

    def delete_by_id(self, ids: Sequence[str], **kwargs: Any) -> SolrUpdateResponse:
        """
        Delete documents from the Solr collection using their IDs.

        If the set of IDs is known, this is generally more efficient than using
        :py:meth:`.delete_by_query`.

        Args:
            ids: A sequence of document IDs to be deleted.
            **kwargs:
                Additional keyword arguments to pass to :py:meth:`pysolr.Solr.delete`.

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
        return self._delete(query_string=None, ids=list(ids), **kwargs)

    def clear_collection(self, **kwargs: Any) -> SolrUpdateResponse:
        """
        Delete all documents from the Solr collection.

        Args:
            **kwargs:
                Optional keyword arguments to be passed to
                :py:meth:`pysolr.Solr.delete`.


        Returns:
            The deserialized response from Solr.

        """
        logger.warning("The Solr collection is being cleared")
        return self.delete_by_query(SolrConstants.QUERY_ALL, **kwargs)
