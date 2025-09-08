#!/usr/bin/env python3
"""
Simple test runner to verify the new SharePointReader features.
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
    print("Testing SharePointReader new features...")

    try:
        from llama_index.readers.microsoft_sharepoint import SharePointReader
        from llama_index.readers.microsoft_sharepoint.event import FileType
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.instrumentation.event_handlers import BaseEventHandler

        print("✓ Successfully imported SharePointReader and events")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    # Dummy credentials for testing
    dummy_kwargs = dict(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
    )

    # Test 1: Custom folder validation
    print("\n1. Testing custom folder validation...")
    try:
        SharePointReader(
            **dummy_kwargs,
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

    # Test 2: Custom parsers with custom folder
    print("\n2. Testing custom parsers with custom folder...")
    try:
        mock_parser = MagicMock()
        reader = SharePointReader(
            **dummy_kwargs,
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

    # Test 3: Custom parsers without custom folder (should use os.getcwd())
    print("\n3. Testing custom parsers without custom folder...")
    try:
        mock_parser = MagicMock()
        reader = SharePointReader(
            **dummy_kwargs,
            custom_parsers={FileType.PDF: mock_parser},
        )
        assert reader.custom_folder == os.getcwd()
        assert reader.custom_parser_manager is not None
        print("✓ Custom parsers without custom folder uses current directory")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 4: Callbacks
    print("\n4. Testing callback functionality...")
    try:
        def document_filter(file_id: str) -> bool:
            return file_id != "skip_me"

        reader = SharePointReader(
            **dummy_kwargs,
            process_document_callback=document_filter,
        )

        assert reader.process_document_callback == document_filter
        assert document_filter("normal_file") is True
        assert document_filter("skip_me") is False

        print("✓ Callbacks work correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 5: Event system
    print("\n5. Testing event system...")
    try:
        reader = SharePointReader(**dummy_kwargs)

        events_received = []

        class TestEventHandler(BaseEventHandler):
            def handle(self, event):
                events_received.append(event.class_name())

        dispatcher = get_dispatcher(__name__)
        event_handler = TestEventHandler()
        dispatcher.add_event_handler(event_handler)

        # Simulate event notification via observer
        from llama_index.readers.microsoft_sharepoint.event import PageDataFetchStartedEvent
        reader._observer.subscribe("PageDataFetchStartedEvent", event_handler.handle)
        event = PageDataFetchStartedEvent(page_id="test_page")
        reader._observer.notify(event)
        assert "PageDataFetchStartedEvent" in events_received

        print("✓ Event system structure is correct")

        # Clean up
        if event_handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(event_handler)
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 6: Error handling
    print("\n6. Testing error handling...")
    try:
        reader1 = SharePointReader(**dummy_kwargs)
        assert reader1.fail_on_error is True  # Default

        reader2 = SharePointReader(**dummy_kwargs, fail_on_error=False)
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