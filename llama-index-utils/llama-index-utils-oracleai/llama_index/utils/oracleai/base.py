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
import traceback
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llama_index.core.schema import Document

if TYPE_CHECKING:
    from oracledb import Connection

logger = logging.getLogger(__name__)


"""OracleSummary class"""


class OracleSummary:
    """Get Summary.

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
        """Get the summary of the input docs.

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
                    """
                    declare
                        input clob;
                    begin
                        input := :data;
                        :summ := dbms_vector_chain.utl_to_summary(input, json(:params));
                    end;""",
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
                    """
                    declare
                        input clob;
                    begin
                        input := :data;
                        :summ := dbms_vector_chain.utl_to_summary(input, json(:params));
                    end;""",
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
                        cursor.execute(
                            """
                            declare
                                input clob;
                            begin
                                input := :data;
                                :summ := dbms_vector_chain.utl_to_summary(input, json(:params));
                            end;""",
                            data=doc,
                            params=json.dumps(self.summary_params),
                            summ=summary,
                        )

                    elif isinstance(doc, Document):
                        cursor.execute(
                            """
                            declare
                                input clob;
                            begin
                                input := :data;
                                :summ := dbms_vector_chain.utl_to_summary(input, json(:params));
                            end;""",
                            data=doc.text,
                            params=json.dumps(self.summary_params),
                            summ=summary,
                        )

                    else:
                        raise Exception("Invalid input type")

                    if summary is None:
                        results.append("")
                    else:
                        results.append(str(summary.getvalue()))

            else:
                raise Exception("Invalid input type")

            cursor.close()
            return results

        except Exception as ex:
            print(f"An exception occurred :: {ex}")
            traceback.print_exc()
            cursor.close()
            raise
