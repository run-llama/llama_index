from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Union
import re
import pandas as pd
import requests
import sqlite3
import tempfile
from llama_index.packs.docugami_kg_rag.config import REPORT_DIRECTORY, DOCUGAMI_API_KEY

from docugami import Docugami
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import BaseTool, QueryEngineTool, ToolMetadata

HEADERS = {"Authorization": f"Bearer {DOCUGAMI_API_KEY}"}


@dataclass
class ReportDetails:
    id: str
    """ID of report."""

    name: str
    """Name of report."""

    local_xlsx_path: Path
    """Local path to XLSX of the report."""

    retrieval_tool_function_name: str
    """Function name for retrieval tool e.g. sql_query_earnings_calls."""

    retrieval_tool_description: str
    """
    Description of retrieval tool e.g. Runs a SQL query over the REPORT_NAME report,
    represented as the following SQL Table... etc."""


def report_name_to_report_query_tool_function_name(name: str) -> str:
    """
    Converts a report name to a report query tool function name.

    Report query tool function names follow these conventions:
    1. Retrieval tool function names always start with "query_".
    2. The rest of the name should be a lowercased string, with underscores for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> report_name_to_report_query_tool_function_name('Earnings Calls')
    'query_earnings_calls'
    >>> report_name_to_report_query_tool_function_name('COVID-19   Statistics')
    'query_covid_19_statistics'
    >>> report_name_to_report_query_tool_function_name('2023 Market Report!!!')
    'query_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"query_{name}"


def report_details_to_report_query_tool_description(name: str, table_info: str) -> str:
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    table_info = re.sub(r"\s+", " ", table_info)
    description = f"Given a single input 'query' parameter, runs a SQL query over the {name} report, represented as the following SQL Table:\n\n{table_info}"

    return description[:2048]  # cap to avoid failures when the description is too long


def download_project_latest_xlsx(project_url: str, local_xlsx: Path) -> Optional[Path]:
    response = requests.get(
        f"{project_url}/artifacts/latest?name=spreadsheet.xlsx",
        headers=HEADERS,
    )

    if response.ok:
        response_json = response.json()["artifacts"]
        xlsx_artifact = next(
            (
                item
                for item in response_json
                if str(item["name"]).lower().endswith(".xlsx")
            ),
            None,
        )
        if xlsx_artifact:
            artifact_id = xlsx_artifact["id"]
            response = requests.get(
                f"{project_url}/artifacts/latest/{artifact_id}/content",
                headers=HEADERS,
            )
            if response.ok:
                os.makedirs(str(local_xlsx.parent), exist_ok=True)
                with open(local_xlsx, "wb") as f:
                    f.write(response.content)
                    return local_xlsx

        raise Exception(f"Failed to download XLSX for {project_url}")
    elif response.status_code == 404:
        return None  # No artifacts found: this project has never been published
    else:
        raise Exception(f"Failed to download XLSX for {project_url}")


def connect_to_excel(
    file_path: Union[Path, str], table_name: str, sample_rows_in_table_info=0
) -> SQLDatabase:
    conn = sqlite3.connect(":memory:")

    file_path = Path(file_path)
    if not (file_path.exists() and file_path.suffix.lower() == ".xlsx"):
        raise Exception(f"Invalid file path: {file_path}")

    df = pd.read_excel(file_path, sheet_name=0)

    df.to_sql(table_name, conn, if_exists="replace", index=False)

    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite")
    with sqlite3.connect(temp_db_file.name) as disk_conn:
        conn.backup(disk_conn)  # dumps the connection to disk

    return SQLDatabase.from_uri(
        f"sqlite:///{temp_db_file.name}",
        sample_rows_in_table_info=sample_rows_in_table_info,
    )


def build_report_details(docset_id: str) -> List[ReportDetails]:
    docugami_client = Docugami()

    projects_response = docugami_client.projects.list()
    if not projects_response or not projects_response.projects:
        return []  # no projects found

    projects = [p for p in projects_response.projects if p.docset.id == docset_id]
    details: List[ReportDetails] = []
    for project in projects:
        local_xlsx_path = download_project_latest_xlsx(
            project.url, Path(REPORT_DIRECTORY) / f"{project.id}.xlsx"
        )
        if local_xlsx_path:
            report_name = project.name or local_xlsx_path.name
            db = connect_to_excel(local_xlsx_path, report_name)
            table_info = db.get_single_table_info(table_name=report_name)
            details.append(
                ReportDetails(
                    id=project.id,
                    name=report_name,
                    local_xlsx_path=local_xlsx_path,
                    retrieval_tool_function_name=report_name_to_report_query_tool_function_name(
                        project.name
                    ),
                    retrieval_tool_description=report_details_to_report_query_tool_description(
                        project.name, table_info
                    ),
                )
            )

    return details


def get_retrieval_tool_for_report(report_details: ReportDetails) -> Optional[BaseTool]:
    if not report_details.local_xlsx_path:
        return None

    db = connect_to_excel(report_details.local_xlsx_path, report_details.name)

    query_engine = NLSQLTableQueryEngine(sql_database=db, tables=[report_details.name])

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=report_details.retrieval_tool_function_name,
            description=report_details.retrieval_tool_description,
        ),
    )
