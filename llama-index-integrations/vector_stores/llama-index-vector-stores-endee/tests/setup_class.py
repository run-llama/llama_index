import os
import time
import unittest
from pathlib import Path
from llama_index.core import Document

# Load .env from project root so ENDEE_API_TOKEN is set when running pytest
def _load_dotenv():
    try:
        from dotenv import load_dotenv
        root = Path(__file__).resolve().parents[1]  # project root
        load_dotenv(root / ".env")
    except ImportError:
        pass  # python-dotenv not installed


_load_dotenv()



try:
    from endee import Endee
except ImportError:
    try:
        from endee.endee import Endee  # fallback
    except ImportError:
        Endee = None


class EndeeTestSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.endee_api_token = os.environ.get("ENDEE_API_TOKEN", "")

        if not cls.endee_api_token:
            raise unittest.SkipTest(
                "ENDEE_API_TOKEN not set. Skipping integration tests."
            )

        if Endee is None:
            raise unittest.SkipTest(
                "endee package not installed. Skipping integration tests."
            )

        cls.nd = Endee(token=cls.endee_api_token)
        # Validate token with a lightweight API call — skip only on auth failure
        try:
            existing = cls.nd.list_indexes()
        except Exception as e:
            raise unittest.SkipTest(
                f"ENDEE_API_TOKEN is invalid or expired, or service is unreachable: {e}. "
                "Update it to run integration tests."
            ) from e

        # Clean up any leftover test indices from previous runs to avoid server memory exhaustion
        for idx in existing.get("indexes", []):
            name = idx.get("name", "")
            if name.startswith("test_index_"):
                try:
                    cls.nd.delete_index(name=name)
                except Exception:
                    pass

        
        # Test index configuration
        timestamp = int(time.time())
        cls.test_index_name = f"test_index_{timestamp}"  # Added timestamp to make unique
        cls.dimension = 384 
        cls.space_type = "cosine"
        
        # Track all test indexes for cleanup
        cls.test_indexes = {cls.test_index_name}
        
        # Create test documents with diverse metadata for comprehensive testing
        cls.test_documents = [
            # Programming category with different languages and difficulties
            Document(
                text="Python is a high-level, interpreted programming language known for its readability and simplicity.",
                metadata={"category": "programming", "language": "python", "difficulty": "beginner"}
            ),
            Document(
                text="JavaScript is a versatile language for web development with advanced features like async/await.",
                metadata={"category": "programming", "language": "javascript", "difficulty": "intermediate"}
            ),
            Document(
                text="Rust provides memory safety without garbage collection using ownership system.",
                metadata={"category": "programming", "language": "rust", "difficulty": "advanced"}
            ),
            
            # AI/ML category with different fields and difficulties
            Document(
                text="Machine learning algorithms learn patterns from data to make predictions.",
                metadata={"category": "ai", "field": "machine_learning", "difficulty": "intermediate"}
            ),
            Document(
                text="Deep learning uses neural networks with multiple layers for complex pattern recognition.",
                metadata={"category": "ai", "field": "deep_learning", "difficulty": "advanced"}
            ),
            
            # Database category with different types and features
            Document(
                text="Vector databases optimize similarity search for high-dimensional data.",
                metadata={"category": "database", "type": "vector", "feature": "similarity_search"}
            ),
            Document(
                text="Time-series databases are optimized for sequential temporal data storage.",
                metadata={"category": "database", "type": "time_series", "feature": "temporal_storage"}
            ),
            
            # Documents with multiple metadata fields for complex filtering
            Document(
                text="Building a real-time ML pipeline with Python and Vector DB.",
                metadata={
                    "category": ["programming", "ai", "database"],
                    "languages": ["python"],
                    "technologies": ["ml", "vector_db"],
                    "difficulty": "advanced"
                }
            ),
            Document(
                text="Implementing secure encryption in distributed databases.",
                metadata={
                    "category": ["programming", "database", "security"],
                    "field": "cryptography",
                    "difficulty": "advanced",
                    "feature": "encryption"
                }
            )
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up by deleting test indexes"""
        # Only run cleanup if we actually set up (i.e., have test_indexes attribute)
        if not hasattr(cls, 'test_indexes'):
            return
        # Wait a moment to allow any pending operations to complete
        time.sleep(2)
        for index_name in cls.test_indexes:
            try:
                cls.nd.delete_index(name=index_name)
                print(f"Successfully deleted test index: {index_name}")
            except Exception as e:
                if "not found" not in str(e).lower():  # Ignore if index doesn't exist
                    print(f"Error deleting test index {index_name}: {e}")

    def setUp(self):
        """Additional setup before each test if needed"""
        pass

    def tearDown(self):
        """Additional cleanup after each test if needed"""
        pass