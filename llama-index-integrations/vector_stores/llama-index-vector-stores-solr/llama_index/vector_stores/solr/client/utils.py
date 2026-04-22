"""Utilities for use with Solr clients, particularly for preparing data for ingestion."""

from datetime import date, datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

from collections.abc import Mapping
from typing import Any, Union, cast

import numpy as np

from llama_index.vector_stores.solr.constants import SolrConstants


def format_datetime_for_solr(dt: Union[datetime, date]) -> str:
    """
    Format an input :py:class:`datetime.datetime` or :py:class:`datetime.date` into a Solr-compatible date string.

    When a timezone is specified (:py:attr:`~datetime.datetime.tzinfo`), it is converted
    to UTC. If one is not specified, it is treated as UTC implicitly.

    See `Solr documentation <https://solr.apache.org/guide/solr/latest/indexing-guide/date-formatting-math.html>`_
    for more information on how Solr treats date fields.

    Examples:
        >>> from datetime import datetime
        >>> from zoneinfo import ZoneInfo
        >>> val = datetime(2025, 2, 18, 1, 2, 3, tzinfo=ZoneInfo("America/New_York"))
        >>> format_datetime_for_solr(val)
        '2025-02-18T06:02:03Z'

    Args:
        dt:
            The input :py:class:`datetime.datetime` or :py:class:`datetime.date`

    Returns:
         A Solr-compatible date string.

    """
    # dates don't have timezones
    if isinstance(dt, datetime):
        if dt.tzinfo is not None:
            # convert other timezone to UTC
            dt = dt.astimezone(UTC)
        else:
            # treat naive datetimes as UTC
            dt = dt.replace(tzinfo=UTC)

    return dt.strftime(SolrConstants.SOLR_ISO8601_DATE_FORMAT)


def prepare_document_for_solr(document: Mapping[str, Any]) -> dict[str, Any]:
    """
    Prepare a document dictionary for insertion into Solr, converting datatypes when necessary.

    The underlying Solr clients used do not always prepare certain datatypes appropriately
    for calls to Solr, which can lead to surprising errors. This function adds some
    special handling to avoid these issues, providing explicit support for the following:

    * :py:class:`bytes` is decoded into a :py:class:`str`
    * :py:class:`datetime.datetime` is formatted into a Solr-compatible date string
    * :py:class:`datetime.date` is formatted into a Solr-compatible date string
    * :py:class:`numpy.ndarray` and its contents are converted into a :py:class:`list`
      of Python primitive types using :py:mod:`numpy` default behavior

    Args:
        document: The document dictionary to prepare.

    Returns:
        A document dictionary prepared for insertion into Solr.

    """
    out_doc: dict[str, Any] = {}
    for key, value in document.items():
        if isinstance(value, (datetime, date)):
            out_doc[key] = format_datetime_for_solr(value)
        elif isinstance(value, np.ndarray):
            out_doc[key] = cast(list, value.tolist())
        elif isinstance(value, np.generic):
            # covers all numpy scalar types, converts to standard python type
            out_doc[key] = value.item()
        elif isinstance(value, bytes):
            out_doc[key] = value.decode()
        else:
            out_doc[key] = value
    return out_doc
