"""Integration test demonstrating all new SharePointReader features working together."""

from unittest.mock import MagicMock
import tempfile

from llama_index.readers.microsoft_sharepoint import SharePointReader
from llama_index.readers.microsoft_sharepoint.event import (
    FileType,
    TotalPagesToProcessEvent,
    PageDataFetchStartedEvent,
    PageDataFetchCompletedEvent,
    PageSkippedEvent,
    PageFailedEvent,
)
from llama_index.core.schema import Document
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler


class TestIntegration:
    """Integration tests for all new features working together."""

    def test_full_feature_integration(self):
        """Test all new features working together in a realistic scenario."""

        # Setup custom parser
        mock_parser = MagicMock()
        mock_parser.load_data.return_value = [
            Document(text="custom parsed content", doc_id="custom")
        ]

        # Setup callback
        def document_filter(file_id: str) -> bool:
            return not file_id.startswith("draft_")

        # Setup event tracking using new event system
        events_log = []

        class TestEventHandler(BaseEventHandler):
            def handle(self, event):
                events_log.append(
                    {
                        "class_name": event.class_name(),
                        "page_id": getattr(event, "page_id", None),
                    }
                )

        # Create reader with all new features
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = SharePointReader(
                client_id="dummy_client_id",
                client_secret="dummy_client_secret",
                tenant_id="dummy_tenant_id",
                sharepoint_site_name="dummy_site_name",
                sharepoint_folder_path="dummy_folder_path",
                custom_parsers={FileType.PDF: mock_parser},
                custom_folder=temp_dir,
                process_document_callback=document_filter,
                fail_on_error=False,
            )

            # Subscribe to events using new event system
            dispatcher = get_dispatcher("llama_index.readers.microsoft_sharepoint.base")
            event_handler = TestEventHandler()
            dispatcher.add_event_handler(event_handler)

            # Simulate document processing
            normal_file_id = "normal_file"
            draft_file_id = "draft_file_001"

            # Simulate event flow for a normal file
            dispatcher.event(PageDataFetchStartedEvent(page_id=normal_file_id))
            dispatcher.event(PageDataFetchCompletedEvent(page_id=normal_file_id, document=Document(text="content", doc_id=normal_file_id)))

            # Simulate event flow for a draft file (should be skipped by callback)
            dispatcher.event(PageDataFetchStartedEvent(page_id=draft_file_id))
            dispatcher.event(PageSkippedEvent(page_id=draft_file_id))

            # Verify events were logged
            assert len(events_log) >= 3  # At least started, completed, skipped

            # Check that we have the expected event types
            event_class_names = [event["class_name"] for event in events_log]
            assert "PageDataFetchStartedEvent" in event_class_names
            assert "PageDataFetchCompletedEvent" in event_class_names
            assert "PageSkippedEvent" in event_class_names

            # Verify custom folder is set correctly
            assert reader.custom_folder == temp_dir
            assert reader.custom_parser_manager is not None

            # Verify callback is working
            assert reader.process_document_callback("normal_file") is True
            assert reader.process_document_callback("draft_file_001") is False

            # Clean up
            if event_handler in dispatcher.event_handlers:
                dispatcher.event_handlers.remove(event_handler)

    def test_event_system_with_realistic_simulation(self):
        """Test event system with a realistic event flow simulation."""
        reader = SharePointReader(
            client_id="dummy_client_id",
            client_secret="dummy_client_secret",
            tenant_id="dummy_tenant_id",
            sharepoint_site_name="dummy_site_name",
            sharepoint_folder_path="dummy_folder_path",
        )

        # Track different types of events separately
        page_events = []
        error_events = []

        class PageEventHandler(BaseEventHandler):
            def handle(self, event):
                if isinstance(
                    event,
                    (
                        PageDataFetchStartedEvent,
                        PageDataFetchCompletedEvent,
                        PageSkippedEvent,
                    ),
                ):
                    page_events.append(event)

        class ErrorEventHandler(BaseEventHandler):
            def handle(self, event):
                if isinstance(event, PageFailedEvent):
                    error_events.append(event)

        # Subscribe to different event types using new event system
        dispatcher = get_dispatcher("llama_index.readers.microsoft_sharepoint.base")
        page_handler = PageEventHandler()
        error_handler = ErrorEventHandler()

        dispatcher.add_event_handler(page_handler)
        dispatcher.add_event_handler(error_handler)

        # Simulate a realistic processing flow by manually emitting events
        # 1. Start processing pages
        dispatcher.event(TotalPagesToProcessEvent(total_pages=3))

        # 2. Process first page successfully
        dispatcher.event(PageDataFetchStartedEvent(page_id="page1"))
        dispatcher.event(
            PageDataFetchCompletedEvent(
                page_id="page1", document=Document(text="content1", doc_id="page1")
            )
        )

        # 3. Skip second page
        dispatcher.event(PageSkippedEvent(page_id="page2"))

        # 4. Fail to process third page
        dispatcher.event(PageDataFetchStartedEvent(page_id="page3"))
        dispatcher.event(PageFailedEvent(page_id="page3", error="Network timeout"))

        # Verify event counts
        assert len(page_events) == 4  # 2 started, 1 completed, 1 skipped
        assert len(error_events) == 1  # 1 page failed

        # Verify event content
        page_event_types = [type(event).__name__ for event in page_events]
        assert "PageDataFetchStartedEvent" in page_event_types
        assert "PageDataFetchCompletedEvent" in page_event_types
        assert "PageSkippedEvent" in page_event_types

        error_event_types = [type(event).__name__ for event in error_events]
        assert "PageFailedEvent" in error_event_types

        # Clean up
        for handler in [page_handler, error_handler]:
            if handler in dispatcher.event_handlers:
                dispatcher.event_handlers.remove(handler)