import unittest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.mint import MintToolSpec


class TestMintToolSpec(unittest.TestCase):
    def test_class_inheritance(self):
        """MintToolSpec should inherit from BaseToolSpec."""
        names_of_base_classes = [b.__name__ for b in MintToolSpec.__mro__]
        self.assertIn(BaseToolSpec.__name__, names_of_base_classes)

    def test_spec_functions(self):
        """All advertised tool functions should be defined on the spec."""
        expected = [
            "attest_work",
            "verify_trust",
            "discover_actors",
            "rate_attestation",
            "recommend_actor",
        ]
        self.assertEqual(MintToolSpec.spec_functions, expected)
        for fn in expected:
            self.assertTrue(callable(getattr(MintToolSpec, fn)))

    def test_initialization(self):
        """The spec should construct a MINT client carrying the given key."""
        tool = MintToolSpec(api_key="fnet_test_key", name="test-agent")
        self.assertEqual(tool.client.api_key, "fnet_test_key")


if __name__ == "__main__":
    unittest.main()
