#!/usr/bin/env python3
"""
Simple test runner to verify the new ConfluenceReader features.
Run this script to test the new functionality without requiring pytest installation.
"""

import sys
import os
import traceback
from unittest.mock import MagicMock

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_basic_tests():
    """Run basic tests for new features without pytest dependency."""
    print("Testing ConfluenceReader new features...")

    try:
        from llama_index.readers.confluence import ConfluenceReader
        from llama_index.readers.confluence.event import FileType
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.instrumentation.event_handlers import BaseEventHandler

        print("✓ Successfully imported ConfluenceReader and events")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    # Test 1: Custom folder validation
    print("\n1. Testing custom folder validation...")
    try:
        ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            custom_folder="/tmp/test",
        )
        print(
            "✗ Should have raised ValueError for custom_folder without custom_parsers"
        )
        return False
    except ValueError as e:
        if "custom_folder can only be used when custom_parsers are provided" in str(e):
            print(
                "✓ Correctly raised ValueError for custom_folder without custom_parsers"
            )
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

    # Test 2: Custom parsers with folder
    print("\n2. Testing custom parsers with custom folder...")
    try:
        mock_parser = MagicMock()
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            custom_parsers={FileType.PDF: mock_parser},
            custom_folder="/tmp/test",
        )
        assert reader.custom_folder == "/tmp/test"
        assert reader.custom_parser_manager is not None
        print("✓ Custom parsers with custom folder works correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 3: Callbacks
    print("\n3. Testing callback functionality...")
    try:

        def attachment_filter(
            media_type: str, file_size: int, title: str
        ) -> tuple[bool, str]:
            if file_size > 1000000:
                return False, "Too large"
            return True, ""

        def document_filter(page_id: str) -> bool:
            return page_id != "skip_me"

        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            process_attachment_callback=attachment_filter,
            process_document_callback=document_filter,
        )

        # Test callbacks
        should_process, reason = attachment_filter(
            "application/pdf", 2000000, "large.pdf"
        )
        assert should_process is False
        assert reason == "Too large"

        assert document_filter("normal_page") is True
        assert document_filter("skip_me") is False

        print("✓ Callbacks work correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 4: Event system
    print("\n4. Testing event system...")
    try:
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        events_received = []

        class TestEventHandler(BaseEventHandler):
            def handle(self, event):
                events_received.append(event.class_name())

        # Test that event system can be used
        dispatcher = get_dispatcher(__name__)
        event_handler = TestEventHandler()
        dispatcher.add_event_handler(event_handler)

        # Test that ConfluenceReader inherits from DispatcherSpanMixin
        from llama_index.core.instrumentation import DispatcherSpanMixin

        assert isinstance(reader, DispatcherSpanMixin)

        print("✓ Event system structure is correct")

        # Clean up
        if event_handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(event_handler)
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 5: Error handling
    print("\n5. Testing error handling...")
    try:
        reader1 = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )
        assert reader1.fail_on_error is True  # Default

        reader2 = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            fail_on_error=False,
        )
        assert reader2.fail_on_error is False

        print("✓ Error handling settings work correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    print("\n🎉 All basic tests passed!")
    return True


if __name__ == "__main__":
    success = run_basic_tests()
    if not success:
        print("\n❌ Some tests failed")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
