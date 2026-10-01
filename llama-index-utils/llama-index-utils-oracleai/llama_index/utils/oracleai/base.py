# -----------------------------------------------------------------------------
# Authors:
#   Harichandan Roy (hroy)
#   David Jiang (ddjiang)
#
# -----------------------------------------------------------------------------
# ...utils/oracleai.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llama_index.core.schema import Document

if TYPE_CHECKING:
    from oracledb import Connection

logger = logging.getLogger(__name__)

# The same anonymous PL/SQL block was previously inlined once per input branch
# (str / Document / List[str] / List[Document]). Hoisting it keeps a single copy
# and lets Oracle reuse one cached statement instead of two indentation variants.
_UTL_TO_SUMMARY_PLSQL = """
declare
    input clob;
begin
    input := :data;
    :summ := dbms_vector_chain.utl_to_summary(input, json(:params));
end;"""


"""OracleSummary class"""


class OracleSummary:
    """
    Get Summary.

    Args:
        conn: Oracle Connection,
        params: Summary parameters,
        proxy: Proxy

    """

    def __init__(
        self, conn: Connection, params: Dict[str, Any], proxy: Optional[str] = None
    ):
        self.conn = conn
        self.proxy = proxy
        self.summary_params = params

    def get_summary(self, docs) -> List[str]:
        """
        Get the summary of the input docs.

        Args:
            docs: The documents to generate summary for.
                  Allowed input types: str, Document, List[str], List[Document]

        Returns:
            List of summary text, one for each input doc.

        """
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        if docs is None:
            return None

        results = []
        cursor = None
        try:
            oracledb.defaults.fetch_lobs = False
            cursor = self.conn.cursor()

            if self.proxy:
                cursor.execute(
                    "begin utl_http.set_proxy(:proxy); end;", proxy=self.proxy
                )

            if isinstance(docs, str):
                results = []

                summary = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.execute(
                    _UTL_TO_SUMMARY_PLSQL,
                    data=docs,
                    params=json.dumps(self.summary_params),
                    summ=summary,
                )

                if summary is None:
                    results.append("")
                else:
                    results.append(str(summary.getvalue()))

            elif isinstance(docs, Document):
                results = []

                summary = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.execute(
                    _UTL_TO_SUMMARY_PLSQL,
                    data=docs.text,
                    params=json.dumps(self.summary_params),
                    summ=summary,
                )

                if summary is None:
                    results.append("")
                else:
                    results.append(str(summary.getvalue()))

            elif isinstance(docs, List):
                results = []
                for doc in docs:
                    summary = cursor.var(oracledb.DB_TYPE_CLOB)
                    if isinstance(doc, str):
                        data = doc
                    elif isinstance(doc, Document):
                        data = doc.text
                    else:
                        raise Exception("Invalid input type")

                    cursor.execute(
                        _UTL_TO_SUMMARY_PLSQL,
                        data=data,
                        params=json.dumps(self.summary_params),
                        summ=summary,
                    )

                    if summary is None:
                        results.append("")
                    else:
                        results.append(str(summary.getvalue()))

            else:
                raise Exception("Invalid input type")

            return results

        except Exception:
            # The previous handler wrote to stdout/stderr via print() and
            # print_exc(), which callers cannot capture or silence through their
            # logging config; the module-level logger above was never used.
            logger.exception("An exception occurred while generating the summary.")
            raise
        finally:
            # cursor.close() used to live in the except branch. If self.conn.cursor()
            # itself raised, cursor was still unbound and close() raised
            # UnboundLocalError, masking the real Oracle error.
            if cursor is not None:
                cursor.close()
