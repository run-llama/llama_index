import time
from collections import namedtuple
from pathlib import Path
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from faker import Faker

    fake = Faker()
except Exception:
    fake = None

MAX_RETRIES = 10
SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL = 0.1
FILING_DETAILS_FILENAME_STEM = "filing-details"
SEC_EDGAR_SEARCH_API_ENDPOINT = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

retries = Retry(
    total=MAX_RETRIES,
    backoff_factor=SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL,
    status_forcelist=[403, 500, 502, 503, 504],
)

FilingMetadata = namedtuple(  # noqa: PYI024
    "FilingMetadata",
    [
        "accession_number",
        "full_submission_url",
        "filing_details_url",
        "filing_details_filename",
    ],
)


class EdgarSearchApiError(Exception):
    pass


def form_request_payload(
    ticker_or_cik: str,
    filing_types: List[str],
    start_date: str,
    end_date: str,
    start_index: int,
    query: str,
) -> dict:
    return {
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "entityName": ticker_or_cik,
        "forms": filing_types,
        "from": start_index,
        "q": query,
    }


def build_filing_metadata_from_hit(hit: dict) -> FilingMetadata:
    accession_number, filing_details_filename = hit["_id"].split(":", 1)
    # Company CIK should be last in the CIK list. This list may also include
    # the CIKs of executives carrying out insider transactions like in form 4.
    cik = hit["_source"]["ciks"][-1]
    accession_number_no_dashes = accession_number.replace("-", "", 2)

    submission_base_url = (
        f"{SEC_EDGAR_ARCHIVES_BASE_URL}/{cik}/{accession_number_no_dashes}"
    )

    full_submission_url = f"{submission_base_url}/{accession_number}.txt"

    # Get XSL if human readable is wanted
    # XSL is required to download the human-readable
    # and styled version of XML documents like form 4
    # SEC_EDGAR_ARCHIVES_BASE_URL + /320193/000032019320000066/wf-form4_159839550969947.xml
    # SEC_EDGAR_ARCHIVES_BASE_URL +
    #           /320193/000032019320000066/xslF345X03/wf-form4_159839550969947.xml

    # xsl = hit["_source"]["xsl"]
    # if xsl is not None:
    #     filing_details_url = f"{submission_base_url}/{xsl}/{filing_details_filename}"
    # else:
    #     filing_details_url = f"{submission_base_url}/{filing_details_filename}"

    filing_details_url = f"{submission_base_url}/{filing_details_filename}"

    filing_details_filename_extension = Path(filing_details_filename).suffix.replace(
        "htm", "html"
    )
    filing_details_filename = (
        f"{FILING_DETAILS_FILENAME_STEM}{filing_details_filename_extension}"
    )

    return FilingMetadata(
        accession_number=accession_number,
        full_submission_url=full_submission_url,
        filing_details_url=filing_details_url,
        filing_details_filename=filing_details_filename,
    )


def generate_random_user_agent() -> str:
    return f"{fake.first_name()} {fake.last_name()} {fake.email()}"


def get_filing_urls_to_download(
    filing_type: str,
    ticker_or_cik: str,
    num_filings_to_download: int,
    after_date: str,
    before_date: str,
    include_amends: bool,
    query: str = "",
) -> List[FilingMetadata]:
    """
    Get the filings URL to download the data.

    Returns:
        List[FilingMetadata]: Filing metadata from SEC

    """
    filings_to_fetch: List[FilingMetadata] = []
    start_index = 0
    client = requests.Session()
    client.mount("http://", HTTPAdapter(max_retries=retries))
    client.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        while len(filings_to_fetch) < num_filings_to_download:
            payload = form_request_payload(
                ticker_or_cik,
                [filing_type],
                after_date,
                before_date,
                start_index,
                query,
            )
            headers = {
                "User-Agent": generate_random_user_agent(),
                "Accept-Encoding": "gzip, deflate",
                "Host": "efts.sec.gov",
            }
            resp = client.post(
                SEC_EDGAR_SEARCH_API_ENDPOINT, json=payload, headers=headers
            )
            resp.raise_for_status()
            search_query_results = resp.json()

            if "error" in search_query_results:
                try:
                    root_cause = search_query_results["error"]["root_cause"]
                    if not root_cause:  # pragma: no cover
                        raise ValueError

                    error_reason = root_cause[0]["reason"]
                    raise EdgarSearchApiError(
                        f"Edgar Search API encountered an error: {error_reason}. "
                        f"Request payload:\n{payload}"
                    )
                except (ValueError, KeyError):  # pragma: no cover
                    raise EdgarSearchApiError(
                        "Edgar Search API encountered an unknown error. "
                        f"Request payload:\n{payload}"
                    ) from None

            query_hits = search_query_results["hits"]["hits"]

            # No more results to process
            if not query_hits:
                break

            for hit in query_hits:
                hit_filing_type = hit["_source"]["file_type"]

                is_amend = hit_filing_type[-2:] == "/A"
                if not include_amends and is_amend:
                    continue
                if is_amend:
                    num_filings_to_download += 1
                # Work around bug where incorrect filings are sometimes included.
                # For example, AAPL 8-K searches include N-Q entries.
                if not is_amend and hit_filing_type != filing_type:
                    continue

                metadata = build_filing_metadata_from_hit(hit)
                filings_to_fetch.append(metadata)

                if len(filings_to_fetch) == num_filings_to_download:
                    return filings_to_fetch

            # Edgar queries 100 entries at a time, but it is best to set this
            # from the response payload in case it changes in the future
            query_size = search_query_results["query"]["size"]
            start_index += query_size

            # Prevent rate limiting
            time.sleep(SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL)
    finally:
        client.close()

    return filings_to_fetch
