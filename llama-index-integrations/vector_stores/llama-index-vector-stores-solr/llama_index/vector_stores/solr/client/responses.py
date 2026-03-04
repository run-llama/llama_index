"""
Pydantic models for Solr responses.

This includes utilities for bridging between responses from ``pysolr`` and ``aiosolr``.
"""

from typing import Any, ClassVar, Optional, Union
from xml.etree import ElementTree as ET

import aiosolr
import pysolr
from pydantic import BaseModel, ConfigDict, Field, alias_generators
from typing_extensions import Self


class SolrResponseHeader(BaseModel):
    """
    Solr response headers.

    The list of fields it not exhaustive, but covers fields present in usage history or
    commonly cited in Solr documentation.

    See `Solr documentation
    <https://solr.apache.org/guide/solr/latest/query-guide/response-writers.html#json-response-writer>`_
    for details.
    """

    status: Optional[int] = Field(default=None)
    """Response status returned by Solr."""

    q_time: Union[float, int, None] = Field(default=None, alias="QTime")
    """Elapsed time (ms) taken by the Solr request handler to complete the request."""

    zk_connected: Optional[bool] = Field(default=None, alias="zkConnected")
    """Optional field indicating whether the request handler was connected to a Zookeeper instance."""

    rf: Optional[int] = Field(default=None)
    """Optional field indicating the number of shards that successfully responded to the request."""

    params: Optional[dict[str, Any]] = Field(default=None)
    """Echoes the request parameters corresponding to the response."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="allow",  # allow extra fields, for forward-compatability
        populate_by_name=True,  # allow both name and alias forms when building
    )


class SolrSelectResponseBody(BaseModel):
    """
    Solr response body.

    See `Solr documentation
    <https://solr.apache.org/guide/solr/latest/query-guide/response-writers.html#json-response-writer>`_
    for details.
    """

    docs: list[dict[str, Any]]
    """Documents returned by Solr for the query.

    Each document is a dictionary containing all of the fields specified in the
    ``fl`` parameter of the request (or a default if not provided).
    """

    num_found: int
    """The number of documents returned by Solr."""

    num_found_exact: bool
    """Whether the ``num_found`` value was approximated or not.

    If ``True``, the real number of hits is guaranteed to be greater than or
    equal to :py:attr:`.num_found`.
    """

    start: int
    """The offset into the query's result set (for paginated queries)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=alias_generators.to_camel,  # generate camelCase aliases
        extra="allow",  # allow extra fields, for forward-compatability
        populate_by_name=True,  # allow both name and alias forms when building
    )


class SolrSelectResponse(BaseModel):
    """
    Solr search response.

    See `Solr documentation
    <https://solr.apache.org/guide/solr/latest/query-guide/response-writers.html#json-response-writer>`_
    for details.
    """

    response: SolrSelectResponseBody
    """The response contents for the input query, containing documents when applicable."""

    response_header: SolrResponseHeader = Field(default_factory=SolrResponseHeader)
    """The header information for the response."""

    debug: Optional[dict[str, Any]] = None
    """Debugging information for the response.

    This will not be present unless indicated in the request.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=alias_generators.to_camel,  # generate camelCase aliases
        extra="allow",  # allow extra fields, for forward-compatability
        populate_by_name=True,  # allow both name and alias forms when building
    )

    @classmethod
    def from_pysolr_results(cls, results: pysolr.Results) -> Self:
        """
        Build a response from a :py:class:`pysolr.Results`.

        This uses the underlying raw response contained in the ``pysolr`` results.
        """
        raw_response: dict[str, Any] = results.raw_response.get("response", {})
        return cls(
            response=SolrSelectResponseBody(
                docs=results.docs,
                num_found=results.hits,
                num_found_exact=raw_response.get("numFoundExact", True),
                start=raw_response.get("start", 0),
            ),
            response_header=results.raw_response.get("responseHeader", {}),
            debug=results.debug,
        )

    @classmethod
    def from_aiosolr_response(cls, results: aiosolr.Response) -> Self:
        """Build a response from a :py:class:`aiosolr.Response`."""
        raw_response: dict[str, Any] = results.data.get("response", {})
        return cls(
            response=SolrSelectResponseBody(
                docs=results.docs,
                num_found=raw_response.get("numFound", 0),
                num_found_exact=raw_response.get("numFoundExact", True),
                start=raw_response.get("start", 0),
            ),
            response_header=SolrResponseHeader(status=results.status),
            debug=results.data.get("debug", {}),
        )


class SolrUpdateResponse(BaseModel):
    """
    Solr update response (add and delete requests).

    See `Solr documentation
    <https://solr.apache.org/guide/solr/latest/query-guide/response-writers.html#json-response-writer>`_
    for details.
    """

    response_header: SolrResponseHeader
    """The header information for the response."""

    debug: Optional[dict[str, Any]] = None
    """Debugging information for the response.

    This will not be present unless indicated in the request.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=alias_generators.to_camel,  # generate camelCase aliases
        extra="allow",  # allow extra fields, for forward-compatability
        populate_by_name=True,  # allow both name and alias forms when building
    )

    @classmethod
    def from_aiosolr_response(cls, results: aiosolr.Response) -> Self:
        """Build an update response from a :py:class:`aiosolr.Response`."""
        return cls(
            response_header=SolrResponseHeader(status=results.status),
            debug=results.data.get("debug", {}),
        )

    @classmethod
    def from_xml(cls, xml: str) -> Self:
        """Parse an update response from return XML."""
        root = ET.fromstring(xml)
        header_data = {}
        header_elem = root.find("./lst[@name='responseHeader']")
        if header_elem is not None:
            for child in header_elem:
                name = child.attrib.get("name")
                header_data[name] = child.text.strip() if child.text else None
        return SolrUpdateResponse(
            response_header=SolrResponseHeader.model_validate(header_data)
        )
