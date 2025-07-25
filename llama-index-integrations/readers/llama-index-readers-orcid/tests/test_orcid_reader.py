import json
import pytest
from unittest.mock import Mock, patch
from requests.exceptions import HTTPError

from llama_index.readers.orcid import ORCIDReader
from llama_index.core.schema import Document


class TestORCIDReader:

    @pytest.fixture
    def mock_profile_data(self):
        return {
            "person": {
                "name": {
                    "given-names": {"value": "Jane"},
                    "family-name": {"value": "Doe"}
                },
                "biography": {
                    "content": "Professor of Computer Science specializing in machine learning."
                },
                "keywords": {
                    "keyword": [
                        {"content": "machine learning"},
                        {"content": "artificial intelligence"}
                    ]
                },
                "external-identifiers": {
                    "external-identifier": [
                        {
                            "external-id-type": "Scopus Author ID",
                            "external-id-value": "12345678900"
                        }
                    ]
                },
                "researcher-urls": {
                    "researcher-url": [
                        {
                            "url-name": "Personal Website",
                            "url": {"value": "https://janedoe.example.com"}
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def mock_works_data(self):
        return {
            "group": [
                {
                    "work-summary": [
                        {
                            "put-code": 12345,
                            "title": {"title": {"value": "Test Publication"}}
                        }
                    ]
                }
            ]
        }

    @pytest.fixture
    def mock_work_detail(self):
        return {
            "title": {"title": {"value": "Machine Learning in Healthcare"}},
            "journal-title": {"value": "Journal of AI Medicine"},
            "publication-date": {"year": {"value": "2023"}},
            "type": "journal-article",
            "url": {"value": "https://example.com/paper"}
        }

    @pytest.fixture
    def mock_employment_data(self):
        return {
            "employment-summary": [
                {
                    "organization": {"name": "Harvard University"},
                    "role-title": "Professor",
                    "department-name": "Computer Science",
                    "start-date": {"year": {"value": "2020"}},
                    "end-date": None
                }
            ]
        }

    @pytest.fixture
    def mock_education_data(self):
        return {
            "education-summary": [
                {
                    "organization": {"name": "MIT"},
                    "role-title": "PhD",
                    "department-name": "Computer Science",
                    "start-date": {"year": {"value": "2015"}},
                    "end-date": {"year": {"value": "2019"}}
                }
            ]
        }

    def test_init_default(self):
        reader = ORCIDReader()
        assert reader.sandbox is False
        assert reader.include_works is True
        assert reader.include_employment is True
        assert reader.include_education is True
        assert reader.max_works == 50
        assert reader.rate_limit_delay == 0.5
        assert reader.base_url == "https://pub.orcid.org/v3.0/"

    def test_init_sandbox(self):
        reader = ORCIDReader(sandbox=True)
        assert reader.sandbox is True
        assert reader.base_url == "https://pub.sandbox.orcid.org/v3.0/"

    def test_validate_orcid_id_valid(self):
        reader = ORCIDReader()
        
        valid_id = "0000-0002-1825-0097"
        assert reader._validate_orcid_id(valid_id) == valid_id
        
        no_hyphens = "0000000218250097"
        expected = "0000-0002-1825-0097"
        assert reader._validate_orcid_id(no_hyphens) == expected
        
        with_url = "https://orcid.org/0000-0002-1825-0097"
        assert reader._validate_orcid_id(with_url) == valid_id
        
        valid_x = "0000-0002-9079-593X"
        assert reader._validate_orcid_id(valid_x) == valid_x

    def test_validate_orcid_id_invalid(self):
        reader = ORCIDReader()
        
        with pytest.raises(ValueError, match="Invalid ORCID ID"):
            reader._validate_orcid_id("invalid-id")
        
        with pytest.raises(ValueError, match="Invalid ORCID ID length"):
            reader._validate_orcid_id("0000-0002-1825")  # Too short
        
        with pytest.raises(ValueError, match="Invalid ORCID ID checksum"):
            reader._validate_orcid_id("0000-0002-1825-0098")  # Wrong checksum
        
        with pytest.raises(ValueError, match="Invalid ORCID ID format"):
            reader._validate_orcid_id("XXXX-0002-1825-0097")  # Non-digit characters

    def test_generate_orcid_checksum(self):
        reader = ORCIDReader()
        
        test_cases = [
            ("000000021825009", "7"),  # 0000-0002-1825-0097
            ("000000020795593", "X"),  # 0000-0002-9079-593X (Stephen Hawking)
            ("000000015551564", "0"),  # 0000-0001-5551-5640
        ]
        
        for base_digits, expected_checksum in test_cases:
            result = reader._generate_orcid_checksum(base_digits)
            assert result == expected_checksum, f"Expected {expected_checksum} for {base_digits}, got {result}"

    @patch('llama_index.readers.orcid.base.requests.Session.get')
    def test_make_request_success(self, mock_get):
        reader = ORCIDReader(rate_limit_delay=0)
        
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = reader._make_request("https://example.com")
        assert result == {"test": "data"}

    @patch('llama_index.readers.orcid.base.requests.Session.get')
    def test_make_request_404(self, mock_get):
        reader = ORCIDReader(rate_limit_delay=0)
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError()
        mock_get.return_value = mock_response
        
        result = reader._make_request("https://example.com")
        assert result is None

    @patch('llama_index.readers.orcid.base.requests.Session.get')
    def test_make_request_rate_limit(self, mock_get):
        reader = ORCIDReader(rate_limit_delay=0)
        
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        error_429 = HTTPError()
        error_429.response = mock_response_429
        mock_response_429.raise_for_status.side_effect = error_429
        
        mock_response_success = Mock()
        mock_response_success.json.return_value = {"test": "data"}
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_429, mock_response_success]
        
        result = reader._make_request("https://example.com")
        assert result == {"test": "data"}
        assert mock_get.call_count == 2

    @patch('llama_index.readers.orcid.base.requests.Session.get')
    def test_make_request_timeout(self, mock_get):
        reader = ORCIDReader(rate_limit_delay=0, timeout=10)
        
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = reader._make_request("https://example.com")
        assert result is None
        assert mock_get.call_count == 4

    @patch('llama_index.readers.orcid.base.requests.Session.get')
    def test_make_request_connection_error(self, mock_get):
        reader = ORCIDReader(rate_limit_delay=0)
        
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = reader._make_request("https://example.com")
        assert result is None
        assert mock_get.call_count == 4

    def test_format_profile_text(self, mock_profile_data):
        reader = ORCIDReader()
        orcid_id = "0000-0002-1825-0097"
        
        result = reader._format_profile_text(mock_profile_data, orcid_id)
        
        assert "ORCID ID: 0000-0002-1825-0097" in result
        assert "Name: Jane Doe" in result
        assert "Professor of Computer Science" in result
        assert "machine learning, artificial intelligence" in result
        assert "Scopus Author ID: 12345678900" in result
        assert "Personal Website: https://janedoe.example.com" in result

    def test_format_works_text(self, mock_work_detail):
        reader = ORCIDReader()
        works_data = {"works": [mock_work_detail]}
        
        result = reader._format_works_text(works_data)
        
        assert "Research Works:" in result
        assert "Machine Learning in Healthcare" in result
        assert "Journal: Journal of AI Medicine" in result
        assert "Year: 2023" in result
        assert "Type: journal-article" in result

    def test_format_affiliation_text(self, mock_employment_data):
        reader = ORCIDReader()
        
        result = reader._format_affiliation_text(mock_employment_data, "Employment")
        
        assert "Employment:" in result
        assert "Harvard University" in result
        assert "Role: Professor" in result
        assert "Department: Computer Science" in result
        assert "From: 2020 to present" in result

    @patch.object(ORCIDReader, '_get_profile_data')
    @patch.object(ORCIDReader, '_get_works_data')
    @patch.object(ORCIDReader, '_get_employment_data')
    @patch.object(ORCIDReader, '_get_education_data')
    def test_load_data_complete(
        self, 
        mock_education, 
        mock_employment, 
        mock_works, 
        mock_profile,
        mock_profile_data,
        mock_employment_data,
        mock_education_data
    ):
        reader = ORCIDReader(rate_limit_delay=0)
        
        mock_profile.return_value = mock_profile_data
        mock_works.return_value = {"works": [{"title": {"title": {"value": "Test Work"}}}]}
        mock_employment.return_value = mock_employment_data
        mock_education.return_value = mock_education_data
        
        documents = reader.load_data(["0000-0002-1825-0097"])
        
        assert len(documents) == 1
        doc = documents[0]
        assert isinstance(doc, Document)
        assert "Jane Doe" in doc.text
        assert "Test Work" in doc.text
        assert "Harvard University" in doc.text
        assert doc.metadata["orcid_id"] == "0000-0002-1825-0097"
        assert doc.metadata["source"] == "ORCID"

    @patch.object(ORCIDReader, '_get_profile_data')
    def test_load_data_no_profile(self, mock_profile):
        reader = ORCIDReader(rate_limit_delay=0)
        mock_profile.return_value = None
        
        documents = reader.load_data(["0000-0002-1825-0097"])
        
        assert len(documents) == 0

    @patch.object(ORCIDReader, '_get_profile_data')
    def test_load_data_multiple_ids(self, mock_profile, mock_profile_data):
        reader = ORCIDReader(rate_limit_delay=0)
        mock_profile.return_value = mock_profile_data
        
        documents = reader.load_data(["0000-0002-1825-0097", "0000-0003-1234-5678"])
        
        assert len(documents) == 2
        assert mock_profile.call_count == 2

    def test_load_data_invalid_orcid_id(self):
        reader = ORCIDReader(rate_limit_delay=0)
        
        documents = reader.load_data(["invalid-id"])
        
        assert len(documents) == 0

    @patch.object(ORCIDReader, '_make_request')
    def test_get_works_data_with_details(self, mock_request, mock_works_data, mock_work_detail):
        reader = ORCIDReader(rate_limit_delay=0, max_works=1)
        
        # First call returns works summary, second call returns work detail
        mock_request.side_effect = [mock_works_data, mock_work_detail]
        
        result = reader._get_works_data("0000-0002-1825-0097")
        
        assert result is not None
        assert "works" in result
        assert len(result["works"]) == 1
        assert mock_request.call_count == 2

    def test_include_flags(self):
        reader = ORCIDReader(
            include_works=False,
            include_employment=False,
            include_education=False,
            rate_limit_delay=0
        )
        
        assert reader._get_works_data("test") is None
        assert reader._get_employment_data("test") is None
        assert reader._get_education_data("test") is None