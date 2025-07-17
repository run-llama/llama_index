"""Integration test demonstrating all new ConfluenceReader features working together."""

from unittest.mock import MagicMock, patch
import tempfile

from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.confluence.event import (
    FileType,
    TotalPagesToProcessEvent,
    PageDataFetchStartedEvent,
    PageDataFetchCompletedEvent,
    PageSkippedEvent,
    PageFailedEvent,
    AttachmentProcessingStartedEvent,
    AttachmentProcessedEvent,
    AttachmentSkippedEvent,
    AttachmentFailedEvent,
)
from llama_index.core.schema import Document
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler


class TestIntegration:
    """Integration tests for all new features working together."""

    @patch("html2text.HTML2Text")
    def test_full_feature_integration(self, mock_html2text_class):
        """Test all new features working together in a realistic scenario."""
        mock_text_maker = MagicMock()
        mock_text_maker.handle.return_value = "processed text content"
        mock_html2text_class.return_value = mock_text_maker

        # Setup custom parser
        mock_parser = MagicMock()
        mock_parser.load_data.return_value = [
            Document(text="custom parsed content", doc_id="custom")
        ]

        # Setup callbacks
        def attachment_filter(
            media_type: str, file_size: int, title: str
        ) -> tuple[bool, str]:
            if "skip" in title.lower():
                return False, "Filename contains 'skip'"
            if file_size > 5000000:  # 5MB
                return False, "File too large"
            return True, ""

        def document_filter(page_id: str) -> bool:
            return not page_id.startswith("draft_")

        # Setup event tracking using new event system
        events_log = []

        class TestEventHandler(BaseEventHandler):
            def handle(self, event):
                events_log.append(
                    {
                        "class_name": event.class_name(),
                        "page_id": getattr(event, "page_id", None),
                        "attachment_name": getattr(event, "attachment_name", None),
                    }
                )

        # Create reader with all new features
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = ConfluenceReader(
                base_url="https://example.atlassian.net/wiki",
                api_token="test_token",
                custom_parsers={FileType.PDF: mock_parser},
                custom_folder=temp_dir,
                process_attachment_callback=attachment_filter,
                process_document_callback=document_filter,
                fail_on_error=False,
            )

            # Subscribe to events using new event system
            dispatcher = get_dispatcher("llama_index.readers.confluence.base")
            event_handler = TestEventHandler()
            dispatcher.add_event_handler(event_handler)

            # Mock confluence client
            reader.confluence = MagicMock()

            # Test document processing
            normal_page = {
                "id": "normal_page",
                "title": "Normal Page",
                "status": "current",
                "body": {"export_view": {"value": "<p>Content</p>"}},
                "_links": {"webui": "/pages/123"},
            }

            draft_page = {
                "id": "draft_page_001",
                "title": "Draft Page",
                "status": "draft",
                "body": {"export_view": {"value": "<p>Draft content</p>"}},
                "_links": {"webui": "/pages/456"},
            }

            # Process normal page (should succeed)
            result1 = reader.process_page(normal_page, False, mock_text_maker)
            assert result1 is not None
            assert result1.doc_id == "normal_page"

            # Process draft page (should be skipped by callback)
            result2 = reader.process_page(draft_page, False, mock_text_maker)
            assert result2 is None  # Skipped by document callback

            # Verify events were logged
            assert len(events_log) >= 2  # At least page started and skipped events

            # Check that we have the expected event types
            event_class_names = [event["class_name"] for event in events_log]
            assert "PageDataFetchStartedEvent" in event_class_names
            assert "PageSkippedEvent" in event_class_names

            # Verify custom folder is set correctly
            assert reader.custom_folder == temp_dir
            assert reader.custom_parser_manager is not None

            # Verify callbacks are working
            should_process, reason = reader.process_attachment_callback(
                "application/pdf", 1000, "normal.pdf"
            )
            assert should_process is True

            should_process, reason = reader.process_attachment_callback(
                "application/pdf", 1000, "skip_this.pdf"
            )
            assert should_process is False
            assert "skip" in reason.lower()

            assert reader.process_document_callback("normal_page") is True
            assert (
                reader.process_document_callback("draft_something") is False
            )  # Clean up
        if event_handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(event_handler)

    def test_event_system_with_realistic_simulation(self):
        """Test event system with a realistic event flow simulation."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        # Track different types of events separately
        page_events = []
        attachment_events = []
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

        class AttachmentEventHandler(BaseEventHandler):
            def handle(self, event):
                if isinstance(
                    event,
                    (
                        AttachmentProcessingStartedEvent,
                        AttachmentProcessedEvent,
                        AttachmentSkippedEvent,
                    ),
                ):
                    attachment_events.append(event)

        class ErrorEventHandler(BaseEventHandler):
            def handle(self, event):
                if isinstance(event, (PageFailedEvent, AttachmentFailedEvent)):
                    error_events.append(event)

        # Subscribe to different event types using new event system
        dispatcher = get_dispatcher("llama_index.readers.confluence.base")
        page_handler = PageEventHandler()
        attachment_handler = AttachmentEventHandler()
        error_handler = ErrorEventHandler()

        dispatcher.add_event_handler(page_handler)
        dispatcher.add_event_handler(attachment_handler)
        dispatcher.add_event_handler(error_handler)

        # Simulate a realistic processing flow by manually emitting events
        # 1. Start processing pages
        dispatcher.event(TotalPagesToProcessEvent(total_pages=3))

        # 2. Process first page successfully
        dispatcher.event(PageDataFetchStartedEvent(page_id="page1"))
        dispatcher.event(
            AttachmentProcessingStartedEvent(
                page_id="page1",
                attachment_id="att1",
                attachment_name="doc1.pdf",
                attachment_type=FileType.PDF,
                attachment_size=1024,
                attachment_link="http://example.com/att1",
            )
        )
        dispatcher.event(
            AttachmentProcessedEvent(
                page_id="page1",
                attachment_id="att1",
                attachment_name="doc1.pdf",
                attachment_type=FileType.PDF,
                attachment_size=1024,
                attachment_link="http://example.com/att1",
            )
        )
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
        assert len(attachment_events) == 2  # 1 started, 1 processed
        assert len(error_events) == 1  # 1 page failed

        # Verify event content
        page_event_types = [type(event).__name__ for event in page_events]
        assert "PageDataFetchStartedEvent" in page_event_types
        assert "PageDataFetchCompletedEvent" in page_event_types
        assert "PageSkippedEvent" in page_event_types

        attachment_event_types = [type(event).__name__ for event in attachment_events]
        assert "AttachmentProcessingStartedEvent" in attachment_event_types
        assert "AttachmentProcessedEvent" in attachment_event_types

        error_event_types = [type(event).__name__ for event in error_events]
        assert "PageFailedEvent" in error_event_types

        # Clean up
        for handler in [page_handler, attachment_handler, error_handler]:
            if handler in dispatcher.event_handlers:
                dispatcher.event_handlers.remove(handler)
