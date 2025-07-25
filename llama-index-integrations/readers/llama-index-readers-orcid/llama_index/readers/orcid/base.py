"""ORCID Reader."""

import logging
import re
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class ORCIDReader(BaseReader):

    def __init__(
        self,
        sandbox: bool = False,
        include_works: bool = True,
        include_employment: bool = True,
        include_education: bool = True,
        max_works: int = 50,
        rate_limit_delay: float = 0.5,
        timeout: int = 30,
    ):
        super().__init__()
        self.sandbox = sandbox
        self.include_works = include_works
        self.include_employment = include_employment
        self.include_education = include_education
        self.max_works = max_works
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        
        if sandbox:
            self.base_url = "https://pub.sandbox.orcid.org/v3.0/"
        else:
            self.base_url = "https://pub.orcid.org/v3.0/"
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "LlamaIndex-ORCID-Reader/1.0",
        })

    def _make_request(self, url: str, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        max_retries = 3
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout for {url} (timeout={self.timeout}s)")
            if retry_count < max_retries:
                logger.info(f"Retrying request (attempt {retry_count + 1}/{max_retries})")
                return self._make_request(url, retry_count + 1)
            else:
                logger.error(f"Max retries exceeded for {url}")
                return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching {url}: {e}")
            if retry_count < max_retries:
                logger.info(f"Retrying after connection error (attempt {retry_count + 1}/{max_retries})")
                time.sleep(2 ** retry_count)
                return self._make_request(url, retry_count + 1)
            else:
                logger.error(f"Max retries exceeded for {url} after connection errors")
                return None
        except requests.exceptions.HTTPError as e:
            response = e.response
            if response and response.status_code == 404:
                logger.warning(f"ORCID profile not found or not public: {url}")
                return None
            elif response and response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting 5 seconds...")
                time.sleep(5)
                return self._make_request(url, retry_count)
            else:
                logger.error(f"HTTP error fetching {url}: {e}")
                return None
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {type(e).__name__}: {e}")
            return None

    def _generate_orcid_checksum(self, base_digits: str) -> str:
        """ISO 7064 MOD 11-2 checksum."""
        total = 0
        for digit in base_digits:
            total = (total + int(digit)) * 2
        
        remainder = total % 11
        result = (12 - remainder) % 11
        
        return 'X' if result == 10 else str(result)

    def _validate_orcid_id(self, orcid_id: str) -> str:
        if "orcid.org/" in orcid_id:
            orcid_id = orcid_id.split("orcid.org/")[-1]
        
        orcid_clean = orcid_id.replace("-", "")
        
        if len(orcid_clean) != 16:
            raise ValueError(f"Invalid ORCID ID length: {orcid_id}")
        
        if not orcid_clean[:15].isdigit():
            raise ValueError(f"Invalid ORCID ID format: {orcid_id}")
        
        if not (orcid_clean[15].isdigit() or orcid_clean[15] == 'X'):
            raise ValueError(f"Invalid ORCID ID checksum character: {orcid_id}")
        
        base_digits = orcid_clean[:15]
        check_digit = orcid_clean[15]
        expected_checksum = self._generate_orcid_checksum(base_digits)
        
        if check_digit != expected_checksum:
            raise ValueError(f"Invalid ORCID ID checksum: {orcid_id}")
        
        if "-" not in orcid_id:
            orcid_id = f"{orcid_clean[:4]}-{orcid_clean[4:8]}-{orcid_clean[8:12]}-{orcid_clean[12:]}"
        
        pattern = re.compile(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$')
        if not pattern.match(orcid_id):
            raise ValueError(f"Invalid ORCID ID format: {orcid_id}")
        
        return orcid_id

    def _get_profile_data(self, orcid_id: str) -> Optional[Dict[str, Any]]:
        url = urljoin(self.base_url, f"{orcid_id}/record")
        return self._make_request(url)

    def _get_works_data(self, orcid_id: str) -> Optional[Dict[str, Any]]:
        if not self.include_works:
            return None
        
        url = urljoin(self.base_url, f"{orcid_id}/works")
        works_summary = self._make_request(url)
        
        if not works_summary or "group" not in works_summary:
            return None
        
        detailed_works = []
        count = 0
        
        for group in works_summary["group"]:
            if count >= self.max_works:
                break
            
            for work_summary in group["work-summary"]:
                if count >= self.max_works:
                    break
                
                put_code = work_summary["put-code"]
                work_url = urljoin(self.base_url, f"{orcid_id}/work/{put_code}")
                work_detail = self._make_request(work_url)
                
                if work_detail:
                    detailed_works.append(work_detail)
                    count += 1
        
        return {"works": detailed_works} if detailed_works else None

    def _get_employment_data(self, orcid_id: str) -> Optional[Dict[str, Any]]:
        if not self.include_employment:
            return None
        
        url = urljoin(self.base_url, f"{orcid_id}/employments")
        return self._make_request(url)

    def _get_education_data(self, orcid_id: str) -> Optional[Dict[str, Any]]:
        if not self.include_education:
            return None
        
        url = urljoin(self.base_url, f"{orcid_id}/educations")
        return self._make_request(url)

    def _format_profile_text(self, profile_data: Dict[str, Any], orcid_id: str) -> str:
        text_parts = [f"ORCID ID: {orcid_id}"]
        
        person = profile_data.get("person", {})
        
        name = person.get("name", {})
        if name:
            given_names = name.get("given-names", {}).get("value", "")
            family_name = name.get("family-name", {}).get("value", "")
            if given_names or family_name:
                text_parts.append(f"Name: {given_names} {family_name}".strip())
        
        biography = person.get("biography", {})
        if biography and biography.get("content"):
            text_parts.append(f"Biography: {biography['content']}")
        
        keywords = person.get("keywords", {}).get("keyword", [])
        if keywords:
            keyword_list = [kw.get("content", "") for kw in keywords if kw.get("content")]
            if keyword_list:
                text_parts.append(f"Keywords: {', '.join(keyword_list)}")
        
        external_ids = person.get("external-identifiers", {}).get("external-identifier", [])
        if external_ids:
            id_list = []
            for ext_id in external_ids:
                id_type = ext_id.get("external-id-type", "")
                id_value = ext_id.get("external-id-value", "")
                if id_type and id_value:
                    id_list.append(f"{id_type}: {id_value}")
            if id_list:
                text_parts.append(f"External IDs: {', '.join(id_list)}")
        
        urls = person.get("researcher-urls", {}).get("researcher-url", [])
        if urls:
            url_list = []
            for url_info in urls:
                name = url_info.get("url-name", "")
                url = url_info.get("url", {}).get("value", "")
                if name and url:
                    url_list.append(f"{name}: {url}")
                elif url:
                    url_list.append(url)
            if url_list:
                text_parts.append(f"URLs: {', '.join(url_list)}")
        
        return "\n".join(text_parts)

    def _format_works_text(self, works_data: Dict[str, Any]) -> str:
        if not works_data or "works" not in works_data:
            return ""
        
        text_parts = ["Research Works:"]
        
        for work in works_data["works"]:
            work_info = work.get("work", work)
            
            title = work_info.get("title", {})
            if title:
                work_title = title.get("title", {}).get("value", "Untitled Work")
                text_parts.append(f"\n• {work_title}")
            
            journal = work_info.get("journal-title", {})
            if journal and journal.get("value"):
                text_parts.append(f"  Journal: {journal['value']}")
            
            pub_date = work_info.get("publication-date")
            if pub_date and pub_date.get("year", {}).get("value"):
                text_parts.append(f"  Year: {pub_date['year']['value']}")
            
            work_type = work_info.get("type")
            if work_type:
                text_parts.append(f"  Type: {work_type}")
            
            url = work_info.get("url", {})
            if url and url.get("value"):
                text_parts.append(f"  URL: {url['value']}")
        
        return "\n".join(text_parts)

    def _format_affiliation_text(self, affiliation_data: Dict[str, Any], section_name: str) -> str:
        if not affiliation_data:
            return ""
        
        text_parts = [f"{section_name}:"]
        
        items_key = "employment-summary" if "employment-summary" in affiliation_data else "education-summary"
        items = affiliation_data.get(items_key, [])
        
        for item in items:
            org = item.get("organization", {})
            org_name = org.get("name", "Unknown Organization")
            
            role = item.get("role-title", "")
            dept = item.get("department-name", "")
            
            text_parts.append(f"\n• {org_name}")
            if role:
                text_parts.append(f"  Role: {role}")
            if dept:
                text_parts.append(f"  Department: {dept}")
            
            start_date = item.get("start-date")
            end_date = item.get("end-date")
            
            if start_date:
                start_year = start_date.get("year", {}).get("value", "")
                date_str = f"  From: {start_year}"
                
                if end_date:
                    end_year = end_date.get("year", {}).get("value", "")
                    date_str += f" to {end_year}"
                else:
                    date_str += " to present"
                
                text_parts.append(date_str)
        
        return "\n".join(text_parts)

    def load_data(self, orcid_ids: List[str]) -> List[Document]:
        """Load researcher profiles from ORCID IDs."""
        documents = []
        
        for orcid_id in orcid_ids:
            try:
                formatted_id = self._validate_orcid_id(orcid_id)
                logger.info(f"Processing ORCID ID: {formatted_id}")
                
                profile_data = self._get_profile_data(formatted_id)
                works_data = self._get_works_data(formatted_id)
                employment_data = self._get_employment_data(formatted_id)
                education_data = self._get_education_data(formatted_id)
                
                if not profile_data:
                    logger.warning(f"No profile data found for ORCID ID: {formatted_id}")
                    continue
                
                profile_text = self._format_profile_text(profile_data, formatted_id)
                
                if works_data:
                    works_text = self._format_works_text(works_data)
                    if works_text:
                        profile_text += f"\n\n{works_text}"
                
                if employment_data:
                    employment_text = self._format_affiliation_text(employment_data, "Employment")
                    if employment_text:
                        profile_text += f"\n\n{employment_text}"
                
                if education_data:
                    education_text = self._format_affiliation_text(education_data, "Education")
                    if education_text:
                        profile_text += f"\n\n{education_text}"
                
                metadata = {
                    "orcid_id": formatted_id,
                    "source": "ORCID",
                    "type": "researcher_profile"
                }
                
                document = Document(text=profile_text, metadata=metadata)
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Error processing ORCID ID {orcid_id}: {e}")
                continue
        
        return documents