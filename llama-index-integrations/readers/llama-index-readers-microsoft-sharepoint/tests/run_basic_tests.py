#!/usr/bin/env python3
"""
Simple test runner to verify the new SharePointReader features.
Run this script to test the new functionality without requiring pytest installation.
"""

import sys
import os
import tempfile
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
        from llama_index.core.readers.base import BaseReader
        from llama_index.core.schema import Document

        print("✓ Successfully imported SharePointReader and events")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    # Dummy credentials for testing
    dummy_kwargs = {
        "client_id": "dummy_client_id",
        "client_secret": "dummy_client_secret",
        "tenant_id": "dummy_tenant_id",
        "sharepoint_site_name": "dummy_site_name",
        "sharepoint_folder_path": "dummy_folder_path",
    }

    # Test 1: Basic class inheritance
    print("\n1. Testing SharePointReader inheritance...")
    try:
        from llama_index.core.readers.base import BasePydanticReader
        from llama_index.core.readers.base import ResourcesReaderMixin
        from llama_index.core.readers import FileSystemReaderMixin
        from llama_index.core.instrumentation import DispatcherSpanMixin

        reader = SharePointReader(**dummy_kwargs)

        # Test inheritance using __mro__ pattern like other tests
        names_of_base_classes = [b.__name__ for b in SharePointReader.__mro__]
        assert BasePydanticReader.__name__ in names_of_base_classes
        assert ResourcesReaderMixin.__name__ in names_of_base_classes
        assert FileSystemReaderMixin.__name__ in names_of_base_classes
        assert DispatcherSpanMixin.__name__ in names_of_base_classes

        print("✓ SharePointReader correctly inherits from all required base classes")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 2: Custom folder validation
    print("\n2. Testing custom folder validation...")
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

    # Test 3: Custom parsers with custom folder
    print("\n3. Testing custom parsers with custom folder...")
    try:
        mock_parser = MagicMock(spec=BaseReader)
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

    # Test 4: Custom parsers without custom folder (should use os.getcwd())
    print("\n4. Testing custom parsers without custom folder...")
    try:
        mock_parser = MagicMock(spec=BaseReader)
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

    # Test 5: Callbacks functionality
    print("\n5. Testing callback functionality...")
    try:

        def document_filter(file_id: str) -> bool:
            return file_id != "skip_me"

        def attachment_filter(media_type: str, file_size: int) -> tuple[bool, str]:
            if file_size > 1000000:
                return False, "File too large"
            return True, ""

        reader = SharePointReader(
            **dummy_kwargs,
            process_document_callback=document_filter,
            process_attachment_callback=attachment_filter,
        )

        assert reader.process_document_callback == document_filter
        assert reader.process_attachment_callback == attachment_filter

        # Test callbacks
        assert document_filter("normal_file") is True
        assert document_filter("skip_me") is False

        should_process, reason = attachment_filter("application/pdf", 2000000)
        assert should_process is False
        assert reason == "File too large"

        print("✓ Callbacks work correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 6: Event system
    print("\n6. Testing event system...")
    try:
        reader = SharePointReader(**dummy_kwargs)

        events_received = []

        class TestEventHandler(BaseEventHandler):
            def handle(self, event):
                events_received.append(event.class_name())

        dispatcher = get_dispatcher(__name__)
        event_handler = TestEventHandler()
        dispatcher.add_event_handler(event_handler)

        # Test event emission patterns
        from llama_index.readers.microsoft_sharepoint.event import (
            PageDataFetchStartedEvent,
            PageDataFetchCompletedEvent,
            PageFailedEvent,
            PageSkippedEvent,
            TotalPagesToProcessEvent,
        )

        # Simulate events - create a proper Document instance for PageDataFetchCompletedEvent
        test_document = Document(text="Test document content", id_="test_doc_1")

        test_events = [
            TotalPagesToProcessEvent(total_pages=5),
            PageDataFetchStartedEvent(page_id="test_page_1"),
            PageDataFetchCompletedEvent(page_id="test_page_1", document=test_document),
            PageSkippedEvent(page_id="test_page_2"),
            PageFailedEvent(page_id="test_page_3", error="Test error"),
        ]

        for event in test_events:
            dispatcher.event(event)

        # Verify events were received
        expected_event_names = [
            "TotalPagesToProcessEvent",
            "PageDataFetchStartedEvent",
            "PageDataFetchCompletedEvent",
            "PageSkippedEvent",
            "PageFailedEvent",
        ]

        assert len(events_received) == len(expected_event_names)
        for expected_name in expected_event_names:
            assert expected_name in events_received

        print("✓ Event system works correctly")

        # Clean up
        if event_handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(event_handler)
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 7: Error handling
    print("\n7. Testing error handling...")
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

    # Test 8: SharePointType enum
    print("\n8. Testing SharePoint type configuration...")
    try:
        from llama_index.readers.microsoft_sharepoint.base import SharePointType

        # Test default type
        reader1 = SharePointReader(**dummy_kwargs)
        assert reader1.sharepoint_type == SharePointType.DRIVE

        # Test explicit type setting
        reader2 = SharePointReader(**dummy_kwargs, sharepoint_type=SharePointType.PAGE)
        assert reader2.sharepoint_type == SharePointType.PAGE

        print("✓ SharePoint type configuration works correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 9: Class name method
    print("\n9. Testing class name method...")
    try:
        assert SharePointReader.class_name() == "SharePointReader"
        print("✓ Class name method works correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 10: File type enum
    print("\n10. Testing FileType enum...")
    try:
        # Test that all expected file types exist
        expected_types = [
            FileType.PDF,
            FileType.HTML,
            FileType.DOCUMENT,
            FileType.PRESENTATION,
            FileType.CSV,
            FileType.SPREADSHEET,
            FileType.IMAGE,
            FileType.JSON,
            FileType.TEXT,
            FileType.TXT,
        ]

        for file_type in expected_types:
            assert isinstance(file_type, FileType)

        print("✓ FileType enum contains all expected types")
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

    # Test 11: Custom parser manager functionality
    print("\n11. Testing CustomParserManager...")
    try:
        from llama_index.readers.microsoft_sharepoint.base import CustomParserManager

        mock_parser = MagicMock(spec=BaseReader)
        mock_parser.load_data.return_value = [MagicMock(text="test content")]

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CustomParserManager(
                custom_parsers={FileType.PDF: mock_parser}, custom_folder=temp_dir
            )

            # Test processing with custom parser
            test_content = b"fake pdf content"
            result = manager.process_with_custom_parser(
                FileType.PDF, test_content, "pdf"
            )

            assert result == "test content"
            mock_parser.load_data.assert_called_once()

        print("✓ CustomParserManager works correctly")
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
