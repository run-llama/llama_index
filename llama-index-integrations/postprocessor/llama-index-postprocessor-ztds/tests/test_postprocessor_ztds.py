"""
Unit tests for LlamaIndex ZTDS Postprocessor
Validates 4 Core Protocol Invariants (IETF draft-sibiryakov-ztds-protocol-02)
"""

import unittest
from ztds_postprocessor import ZTDSNodePostprocessor


class MockTextNode:
    def __init__(self, text: str):
        self.text = text

    def get_content(self) -> str:
        return self.text

    def set_content(self, text: str) -> None:
        self.text = text


class MockNodeWithScore:
    def __init__(self, text: str, score: float = 0.95):
        self.node = MockTextNode(text)
        self.score = score


class TestLlamaIndexZTDSPostprocessor(unittest.TestCase):
    def setUp(self):
        self.postprocessor = ZTDSNodePostprocessor(session_id="llama-test-01")

    def test_node_sanitization(self):
        mock_ssn = "-".join(["123", "45", "6789"])
        raw_content = f"Patient record: Dr. John Doe (john.doe@clinic.org) prescribed treatment for patient SSN {mock_ssn}."
        nodes = [MockNodeWithScore(raw_content)]

        processed_nodes = self.postprocessor._postprocess_nodes(nodes)
        sanitized_content = processed_nodes[0].node.get_content()

        self.assertNotIn("john.doe@clinic.org", sanitized_content)
        self.assertNotIn(mock_ssn, sanitized_content)
        self.assertIn("[EMAIL_TOKEN_1]", sanitized_content)
        self.assertIn("[SSN_TOKEN_1]", sanitized_content)

        # Restore test
        restored = self.postprocessor.restore_text(sanitized_content)
        self.assertEqual(restored, raw_content)

        # Invariant 3: RAM Zeroization
        self.postprocessor.zeroize()
        self.assertNotIn("llama-test-01", self.postprocessor._session_maps)


if __name__ == "__main__":
    unittest.main()
