"""
SEC Filings Streaming Reader.

A streaming reader that fetches SEC filings directly without local file storage,
supports 8-K filings, and provides structured section extraction with rich metadata.
"""

import concurrent.futures
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

try:
    import requests
    from ratelimit import limits, sleep_and_retry
except ImportError:
    requests = None

    def fake_decorator(*args, **kwargs):
        def inner(func):
            return func

        return inner

    limits = fake_decorator
    sleep_and_retry = fake_decorator


# SEC API endpoints
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_EDGAR_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_EDGAR_SEARCH_API_ENDPOINT = "https://efts.sec.gov/LATEST/search-index"


# 8-K Item definitions
SECTIONS_8K = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation or Off-Balance Sheet Arrangement",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendment to Registrant's Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Nominations Pursuant to Exchange Act Rule 14a-11",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or Other External Support",
    "6.04": "Failure to Make a Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

# Section patterns for 10-K
SECTIONS_10K = {
    "ITEM_1": re.compile(r"(?i)item\s*1[.:]?\s*business", re.IGNORECASE),
    "ITEM_1A": re.compile(r"(?i)item\s*1a[.:]?\s*risk\s*factors", re.IGNORECASE),
    "ITEM_1B": re.compile(
        r"(?i)item\s*1b[.:]?\s*unresolved\s*staff\s*comments", re.IGNORECASE
    ),
    "ITEM_2": re.compile(r"(?i)item\s*2[.:]?\s*properties", re.IGNORECASE),
    "ITEM_3": re.compile(r"(?i)item\s*3[.:]?\s*legal\s*proceedings", re.IGNORECASE),
    "ITEM_4": re.compile(r"(?i)item\s*4[.:]?\s*mine\s*safety", re.IGNORECASE),
    "ITEM_5": re.compile(
        r"(?i)item\s*5[.:]?\s*market\s*for\s*(the\s*)?registrant", re.IGNORECASE
    ),
    "ITEM_6": re.compile(r"(?i)item\s*6[.:]?\s*(reserved|\[reserved\])", re.IGNORECASE),
    "ITEM_7": re.compile(r"(?i)item\s*7[.:]?\s*management.*discussion", re.IGNORECASE),
    "ITEM_7A": re.compile(
        r"(?i)item\s*7a[.:]?\s*(quantitative|qualitative).*market\s*risk",
        re.IGNORECASE,
    ),
    "ITEM_8": re.compile(r"(?i)item\s*8[.:]?\s*financial\s*statements", re.IGNORECASE),
    "ITEM_9": re.compile(
        r"(?i)item\s*9[.:]?\s*changes.*disagreements.*accountants", re.IGNORECASE
    ),
    "ITEM_9A": re.compile(
        r"(?i)item\s*9a[.:]?\s*controls\s*and\s*procedures", re.IGNORECASE
    ),
    "ITEM_9B": re.compile(r"(?i)item\s*9b[.:]?\s*other\s*information", re.IGNORECASE),
    "ITEM_10": re.compile(r"(?i)item\s*10[.:]?\s*directors.*officers", re.IGNORECASE),
    "ITEM_11": re.compile(
        r"(?i)item\s*11[.:]?\s*executive\s*compensation", re.IGNORECASE
    ),
    "ITEM_12": re.compile(r"(?i)item\s*12[.:]?\s*security\s*ownership", re.IGNORECASE),
    "ITEM_13": re.compile(
        r"(?i)item\s*13[.:]?\s*certain\s*relationships", re.IGNORECASE
    ),
    "ITEM_14": re.compile(r"(?i)item\s*14[.:]?\s*principal\s*account", re.IGNORECASE),
    "ITEM_15": re.compile(r"(?i)item\s*15[.:]?\s*exhibits", re.IGNORECASE),
}

# Section patterns for 10-Q
SECTIONS_10Q = {
    "PART_I_ITEM_1": re.compile(
        r"(?i)(part\s*i.*)?item\s*1[.:]?\s*financial\s*statements", re.IGNORECASE
    ),
    "PART_I_ITEM_2": re.compile(
        r"(?i)(part\s*i.*)?item\s*2[.:]?\s*management.*discussion", re.IGNORECASE
    ),
    "PART_I_ITEM_3": re.compile(
        r"(?i)(part\s*i.*)?item\s*3[.:]?\s*(quantitative|qualitative).*market",
        re.IGNORECASE,
    ),
    "PART_I_ITEM_4": re.compile(
        r"(?i)(part\s*i.*)?item\s*4[.:]?\s*controls", re.IGNORECASE
    ),
    "PART_II_ITEM_1": re.compile(
        r"(?i)(part\s*ii.*)?item\s*1[.:]?\s*legal\s*proceedings", re.IGNORECASE
    ),
    "PART_II_ITEM_1A": re.compile(
        r"(?i)(part\s*ii.*)?item\s*1a[.:]?\s*risk\s*factors", re.IGNORECASE
    ),
    "PART_II_ITEM_2": re.compile(
        r"(?i)(part\s*ii.*)?item\s*2[.:]?\s*unregistered\s*sales", re.IGNORECASE
    ),
    "PART_II_ITEM_3": re.compile(
        r"(?i)(part\s*ii.*)?item\s*3[.:]?\s*defaults", re.IGNORECASE
    ),
    "PART_II_ITEM_4": re.compile(
        r"(?i)(part\s*ii.*)?item\s*4[.:]?\s*mine\s*safety", re.IGNORECASE
    ),
    "PART_II_ITEM_5": re.compile(
        r"(?i)(part\s*ii.*)?item\s*5[.:]?\s*other\s*information", re.IGNORECASE
    ),
    "PART_II_ITEM_6": re.compile(
        r"(?i)(part\s*ii.*)?item\s*6[.:]?\s*exhibits", re.IGNORECASE
    ),
}


@dataclass
class FilingMetadata:
    """Metadata for an SEC filing."""

    ticker: str
    cik: str
    company_name: str
    filing_type: str
    filing_date: str
    accession_number: str
    primary_document: str
    filing_url: str
    description: Optional[str] = None


class SECFilingsStreamingReader(BaseReader):
    """
    SEC Filings Streaming Reader.

    A streaming reader that fetches SEC filings directly without local file storage.
    Supports 10-K, 10-Q, and 8-K filings with structured section extraction and
    rich metadata.

    Features:
    - Direct streaming: No local file storage required
    - 8-K support: Full support for 8-K current report filings
    - Structured sections: Extract specific sections (Item 1A Risk Factors, etc.)
    - Rich metadata: CIK, filing date, accession number, company name

    Examples:
        >>> from llama_index.readers.sec_filings import SECFilingsStreamingReader
        >>>
        >>> # Basic usage - get all documents
        >>> reader = SECFilingsStreamingReader(
        ...     tickers=["AAPL", "MSFT"],
        ...     filing_types=["10-K", "8-K"],
        ...     num_filings=5,
        ... )
        >>> documents = reader.load_data()
        >>>
        >>> # Get specific sections only
        >>> reader = SECFilingsStreamingReader(
        ...     tickers=["AAPL"],
        ...     filing_types=["10-K"],
        ...     num_filings=3,
        ...     sections=["ITEM_1A", "ITEM_7"],  # Risk Factors and MD&A
        ... )
        >>> documents = reader.load_data()
        >>>
        >>> # Access rich metadata
        >>> for doc in documents:
        ...     print(f"Company: {doc.metadata['company_name']}")
        ...     print(f"CIK: {doc.metadata['cik']}")
        ...     print(f"Filing Date: {doc.metadata['filing_date']}")
        ...     print(f"Accession Number: {doc.metadata['accession_number']}")

    """

    def __init__(
        self,
        tickers: List[str],
        filing_types: Optional[List[str]] = None,
        num_filings: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sections: Optional[List[str]] = None,
        include_amends: bool = False,
        company: Optional[str] = None,
        email: Optional[str] = None,
        num_workers: int = 4,
    ):
        """
        Initialize the SEC Filings Streaming Reader.

        Args:
            tickers: List of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
            filing_types: List of filing types to fetch. Supported: "10-K", "10-Q", "8-K".
                Defaults to ["10-K"].
            num_filings: Number of filings to fetch per ticker per filing type.
            start_date: Start date for filing search (format: "YYYY-MM-DD").
            end_date: End date for filing search (format: "YYYY-MM-DD").
            sections: List of specific sections to extract. If None, extracts full document.
                For 10-K: "ITEM_1", "ITEM_1A", "ITEM_7", etc.
                For 10-Q: "PART_I_ITEM_1", "PART_I_ITEM_2", etc.
                For 8-K: "1.01", "2.02", "7.01", "8.01", etc.
            include_amends: Whether to include amended filings (e.g., 10-K/A).
            company: Company name for SEC API user agent (required by SEC).
            email: Email for SEC API user agent (required by SEC).
            num_workers: Number of concurrent workers for fetching filings.

        Raises:
            ValueError: If invalid filing type is provided.
            ImportError: If required dependencies are not installed.

        """
        if requests is None:
            raise ImportError(
                "Please install requests to use SECFilingsStreamingReader. "
                "You can do so by running `pip install requests`."
            )

        self.tickers = [t.upper() for t in tickers]
        self.filing_types = filing_types or ["10-K"]
        self.num_filings = num_filings
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.sections = sections
        self.include_amends = include_amends
        self.company = company or "LlamaIndex"
        self.email = email or "support@llamaindex.ai"
        self.num_workers = num_workers

        # Validate filing types
        valid_types = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A"}
        for ft in self.filing_types:
            if ft not in valid_types:
                raise ValueError(
                    f"Invalid filing type: {ft}. Supported types: {valid_types}"
                )

        self._session = None

    def _get_session(self) -> requests.Session:
        """Get or create a requests session with proper headers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": f"{self.company} {self.email}",
                    "Accept-Encoding": "gzip, deflate",
                }
            )
        return self._session

    @sleep_and_retry
    @limits(calls=10, period=1)
    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from URL with rate limiting."""
        try:
            response = self._get_session().get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number from ticker symbol."""
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=include&count=1&output=atom"
        response = self._fetch_url(url)
        if response:
            # Extract CIK from response
            cik_match = re.search(r"CIK=(\d{10})", response)
            if cik_match:
                return cik_match.group(1)
        return None

    def _get_company_info(self, cik: str) -> Dict[str, Any]:
        """Get company information from SEC."""
        url = f"{SEC_SUBMISSIONS_URL}/CIK{cik.zfill(10)}.json"
        response = self._fetch_url(url)
        if response:
            import json

            return json.loads(response)
        return {}

    def _search_filings(self, ticker: str, filing_type: str) -> List[FilingMetadata]:
        """Search for filings using SEC EDGAR full-text search API."""
        filings = []

        # Get CIK first
        cik = self._get_cik_by_ticker(ticker)
        if not cik:
            print(f"Could not find CIK for ticker {ticker}")
            return filings

        # Get company info
        company_info = self._get_company_info(cik)
        company_name = company_info.get("name", ticker)

        recent_filings = company_info.get("filings", {}).get("recent", {})
        if not recent_filings:
            return filings

        form_types = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        primary_documents = recent_filings.get("primaryDocument", [])
        descriptions = recent_filings.get("primaryDocDescription", [])

        count = 0
        for i, form in enumerate(form_types):
            if count >= self.num_filings:
                break

            # Check if this is the filing type we want
            if self.include_amends:
                if not (form == filing_type or form == f"{filing_type}/A"):
                    continue
            else:
                if form != filing_type:
                    continue

            # Check date range
            filing_date = filing_dates[i] if i < len(filing_dates) else ""
            if self.start_date and filing_date < self.start_date:
                continue
            if self.end_date and filing_date > self.end_date:
                continue

            accession = accession_numbers[i] if i < len(accession_numbers) else ""
            primary_doc = primary_documents[i] if i < len(primary_documents) else ""
            description = descriptions[i] if i < len(descriptions) else ""

            accession_no_dashes = accession.replace("-", "")
            filing_url = f"{SEC_EDGAR_ARCHIVES_BASE_URL}/{cik}/{accession_no_dashes}/{primary_doc}"

            filings.append(
                FilingMetadata(
                    ticker=ticker,
                    cik=cik,
                    company_name=company_name,
                    filing_type=form,
                    filing_date=filing_date,
                    accession_number=accession,
                    primary_document=primary_doc,
                    filing_url=filing_url,
                    description=description,
                )
            )
            count += 1

        return filings

    def _extract_sections_10k(self, text: str, sections: List[str]) -> Dict[str, str]:
        """Extract specific sections from a 10-K filing."""
        results = {}
        text_lower = text.lower()

        for section_name in sections:
            if section_name not in SECTIONS_10K:
                continue

            pattern = SECTIONS_10K[section_name]
            matches = list(pattern.finditer(text))

            if not matches:
                continue

            # Find the start of the section
            start_match = matches[0]
            start_pos = start_match.end()

            # Find the end (next section or end of document)
            end_pos = len(text)
            for next_section, next_pattern in SECTIONS_10K.items():
                if next_section <= section_name:
                    continue
                next_matches = list(next_pattern.finditer(text[start_pos:]))
                if next_matches:
                    end_pos = min(end_pos, start_pos + next_matches[0].start())

            section_text = text[start_pos:end_pos].strip()
            if section_text:
                results[section_name] = section_text

        return results

    def _extract_sections_10q(self, text: str, sections: List[str]) -> Dict[str, str]:
        """Extract specific sections from a 10-Q filing."""
        results = {}

        for section_name in sections:
            if section_name not in SECTIONS_10Q:
                continue

            pattern = SECTIONS_10Q[section_name]
            matches = list(pattern.finditer(text))

            if not matches:
                continue

            start_match = matches[0]
            start_pos = start_match.end()

            end_pos = len(text)
            for next_section, next_pattern in SECTIONS_10Q.items():
                if next_section <= section_name:
                    continue
                next_matches = list(next_pattern.finditer(text[start_pos:]))
                if next_matches:
                    end_pos = min(end_pos, start_pos + next_matches[0].start())

            section_text = text[start_pos:end_pos].strip()
            if section_text:
                results[section_name] = section_text

        return results

    def _extract_sections_8k(self, text: str, sections: List[str]) -> Dict[str, str]:
        """Extract specific sections from an 8-K filing."""
        results = {}

        for section_name in sections:
            if section_name not in SECTIONS_8K:
                continue

            # Pattern for 8-K items: "Item X.XX"
            pattern = re.compile(
                rf"(?i)item\s*{re.escape(section_name)}[.:]?\s*{re.escape(SECTIONS_8K[section_name])}",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(text))

            if not matches:
                # Try simpler pattern
                pattern = re.compile(
                    rf"(?i)item\s*{re.escape(section_name)}", re.IGNORECASE
                )
                matches = list(pattern.finditer(text))

            if not matches:
                continue

            start_match = matches[0]
            start_pos = start_match.end()

            # Find next item
            next_item_pattern = re.compile(r"(?i)item\s*\d+\.\d+", re.IGNORECASE)
            next_matches = list(next_item_pattern.finditer(text[start_pos:]))

            if next_matches:
                end_pos = start_pos + next_matches[0].start()
            else:
                end_pos = len(text)

            section_text = text[start_pos:end_pos].strip()
            if section_text:
                results[section_name] = section_text

        return results

    def _clean_html(self, html_text: str) -> str:
        """Clean HTML content and extract text."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html_text)
        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Remove non-breaking spaces
        text = text.replace("&nbsp;", " ")
        text = text.replace("&#160;", " ")
        # Decode common HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        return text.strip()

    def _process_filing(self, metadata: FilingMetadata) -> List[Document]:
        """Process a single filing and return documents."""
        documents = []

        # Fetch the filing content
        content = self._fetch_url(metadata.filing_url)
        if not content:
            return documents

        # Clean HTML
        text = self._clean_html(content)

        # Base metadata
        base_metadata = {
            "ticker": metadata.ticker,
            "cik": metadata.cik,
            "company_name": metadata.company_name,
            "filing_type": metadata.filing_type,
            "filing_date": metadata.filing_date,
            "accession_number": metadata.accession_number,
            "filing_url": metadata.filing_url,
            "description": metadata.description,
        }

        # If no specific sections requested, return full document
        if not self.sections:
            documents.append(
                Document(
                    text=text,
                    metadata={
                        **base_metadata,
                        "section": "FULL_DOCUMENT",
                    },
                )
            )
            return documents

        # Extract sections based on filing type
        filing_type_base = metadata.filing_type.replace("/A", "")

        if filing_type_base == "10-K":
            sections = self._extract_sections_10k(text, self.sections)
        elif filing_type_base == "10-Q":
            sections = self._extract_sections_10q(text, self.sections)
        elif filing_type_base == "8-K":
            sections = self._extract_sections_8k(text, self.sections)
        else:
            sections = {}

        # Create documents for each section
        for section_name, section_text in sections.items():
            documents.append(
                Document(
                    text=section_text,
                    metadata={
                        **base_metadata,
                        "section": section_name,
                    },
                )
            )

        return documents

    def load_data(self) -> List[Document]:
        """
        Load SEC filings data.

        Returns a list of Documents containing the filing content with rich metadata.
        Each document includes:
        - ticker: Stock ticker symbol
        - cik: Central Index Key
        - company_name: Company name
        - filing_type: Type of filing (10-K, 10-Q, 8-K)
        - filing_date: Date of filing
        - accession_number: SEC accession number
        - filing_url: URL to the filing
        - section: Section name (if sections were specified)

        Returns:
            List of Document objects.

        """
        all_documents = []
        all_filings = []

        # Collect all filings to process
        for ticker in self.tickers:
            for filing_type in self.filing_types:
                filings = self._search_filings(ticker, filing_type)
                all_filings.extend(filings)
                print(f"Found {len(filings)} {filing_type} filings for {ticker}")

        # Process filings with thread pool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            future_to_filing = {
                executor.submit(self._process_filing, filing): filing
                for filing in all_filings
            }

            for future in concurrent.futures.as_completed(future_to_filing):
                filing = future_to_filing[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    print(
                        f"Processed {filing.filing_type} for {filing.ticker} "
                        f"({filing.filing_date}): {len(documents)} documents"
                    )
                except Exception as e:
                    print(
                        f"Error processing {filing.filing_type} for {filing.ticker}: {e}"
                    )

        return all_documents
