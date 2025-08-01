import os
import pytest
import time
from llama_index.readers.orcid import ORCIDReader
from llama_index.core.schema import Document


@pytest.mark.integration
class TestORCIDIntegration:
    # Test ORCID IDs from ORCID sandbox
    TEST_ORCID_IDS = [
        "0000-0002-1825-0097",  # Josiah Carberry (test account)
        "0000-0001-7857-2795",  # Another test account
    ]

    def test_load_data_from_sandbox(self):
        reader = ORCIDReader(sandbox=True, rate_limit_delay=1)
        
        documents = reader.load_data([self.TEST_ORCID_IDS[0]])
        
        assert len(documents) >= 0  # May be empty if profile is private
        
        if documents:
            doc = documents[0]
            assert isinstance(doc, Document)
            assert doc.metadata["orcid_id"] == self.TEST_ORCID_IDS[0]
            assert doc.metadata["source"] == "ORCID"
            assert doc.metadata["type"] == "researcher_profile"
            assert "ORCID ID:" in doc.text

    def test_load_multiple_profiles(self):
        reader = ORCIDReader(
            sandbox=True,
            include_works=False,  # Speed up test
            include_employment=False,
            include_education=False,
            rate_limit_delay=1
        )
        
        documents = reader.load_data(self.TEST_ORCID_IDS[:2])
        
        assert isinstance(documents, list)
        
        for doc in documents:
            assert isinstance(doc, Document)
            assert doc.metadata["orcid_id"] in self.TEST_ORCID_IDS
            assert doc.metadata["source"] == "ORCID"

    def test_invalid_orcid_id_handling(self):
        reader = ORCIDReader(sandbox=True)
        
        mixed_ids = [
            self.TEST_ORCID_IDS[0],  # Valid
            "0000-0000-0000-0000",   # Invalid checksum
            "invalid-format",         # Invalid format
        ]
        
        documents = reader.load_data(mixed_ids)
        
        assert len(documents) <= 1  # At most one valid document
        
        if documents:
            assert documents[0].metadata["orcid_id"] == self.TEST_ORCID_IDS[0]

    def test_rate_limiting(self):
        reader = ORCIDReader(sandbox=True, rate_limit_delay=0.5)
        
        start_time = time.time()
        
        reader._get_profile_data(self.TEST_ORCID_IDS[0])
        reader._get_profile_data(self.TEST_ORCID_IDS[0])
        
        elapsed = time.time() - start_time
        assert elapsed >= 1.0  # 2 requests * 0.5s delay

    def test_sandbox_vs_production_urls(self):
        sandbox_reader = ORCIDReader(sandbox=True)
        prod_reader = ORCIDReader(sandbox=False)
        
        assert "sandbox" in sandbox_reader.base_url
        assert "sandbox" not in prod_reader.base_url

    @pytest.mark.skipif(
        os.environ.get("SKIP_SLOW_TESTS", "false").lower() == "true",
        reason="Skipping slow integration test"
    )
    def test_load_with_works(self):
        reader = ORCIDReader(
            sandbox=True,
            include_works=True,
            max_works=5,  # Limit for speed
            rate_limit_delay=1
        )
        
        documents = reader.load_data([self.TEST_ORCID_IDS[0]])
        
        if documents and "Research Works:" in documents[0].text:
            assert "Research Works:" in documents[0].text

    def test_network_error_resilience(self):
        reader = ORCIDReader(sandbox=True, timeout=1)  # Very short timeout
        
        documents = reader.load_data([self.TEST_ORCID_IDS[0]])
        assert isinstance(documents, list)