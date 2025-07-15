"""Integration test demonstrating all new ConfluenceReader features working together."""

from unittest.mock import MagicMock, patch
import tempfile

from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.confluence.event import EventName, FileType
from llama_index.core.schema import Document


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

        # Setup event tracking
        events_log = []

        def track_all_events(event):
            events_log.append(
                {
                    "name": event.name,
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

            # Subscribe to events
            reader.observer.subscribe_all(track_all_events)

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
            event_names = [event["name"] for event in events_log]
            assert EventName.PAGE_DATA_FETCH_STARTED in event_names
            assert EventName.PAGE_SKIPPED in event_names

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
            assert reader.process_document_callback("draft_something") is False

    def test_observer_with_real_events_simulation(self):
        """Test observer pattern with a realistic event flow simulation."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        # Track different types of events separately
        page_events = []
        attachment_events = []
        error_events = []

        def handle_page_events(event):
            page_events.append(event)

        def handle_attachment_events(event):
            attachment_events.append(event)

        def handle_error_events(event):
            error_events.append(event)

        # Subscribe to different event types
        reader.observer.subscribe(EventName.PAGE_DATA_FETCH_STARTED, handle_page_events)
        reader.observer.subscribe(
            EventName.PAGE_DATA_FETCH_COMPLETED, handle_page_events
        )
        reader.observer.subscribe(EventName.PAGE_SKIPPED, handle_page_events)

        reader.observer.subscribe(
            EventName.ATTACHMENT_PROCESSING_STARTED, handle_attachment_events
        )
        reader.observer.subscribe(
            EventName.ATTACHMENT_PROCESSED, handle_attachment_events
        )
        reader.observer.subscribe(
            EventName.ATTACHMENT_SKIPPED, handle_attachment_events
        )

        reader.observer.subscribe(EventName.PAGE_FAILED, handle_error_events)
        reader.observer.subscribe(EventName.ATTACHMENT_FAILED, handle_error_events)

        # Simulate a realistic processing flow
        from llama_index.readers.confluence.event import PageEvent, AttachmentEvent

        # 1. Start processing pages
        reader.observer.notify(
            PageEvent(
                name=EventName.TOTAL_PAGES_TO_PROCESS,
                page_id="",
                document=Document(text="", doc_id=""),
                metadata={"total_pages": 3},
            )
        )

        # 2. Process first page successfully
        reader.observer.notify(
            PageEvent(
                name=EventName.PAGE_DATA_FETCH_STARTED,
                page_id="page1",
                document=Document(text="content1", doc_id="page1"),
            )
        )

        reader.observer.notify(
            AttachmentEvent(
                name=EventName.ATTACHMENT_PROCESSING_STARTED,
                page_id="page1",
                attachment_id="att1",
                attachment_name="doc1.pdf",
                attachment_type="application/pdf",
                attachment_size=1024,
                attachment_link="http://example.com/att1",
            )
        )

        reader.observer.notify(
            AttachmentEvent(
                name=EventName.ATTACHMENT_PROCESSED,
                page_id="page1",
                attachment_id="att1",
                attachment_name="doc1.pdf",
                attachment_type="application/pdf",
                attachment_size=1024,
                attachment_link="http://example.com/att1",
            )
        )

        reader.observer.notify(
            PageEvent(
                name=EventName.PAGE_DATA_FETCH_COMPLETED,
                page_id="page1",
                document=Document(text="content1", doc_id="page1"),
            )
        )

        # 3. Skip second page
        reader.observer.notify(
            PageEvent(
                name=EventName.PAGE_SKIPPED,
                page_id="page2",
                document=Document(text="", doc_id="page2"),
            )
        )

        # 4. Fail to process third page
        reader.observer.notify(
            PageEvent(
                name=EventName.PAGE_DATA_FETCH_STARTED,
                page_id="page3",
                document=Document(text="content3", doc_id="page3"),
            )
        )

        reader.observer.notify(
            PageEvent(
                name=EventName.PAGE_FAILED,
                page_id="page3",
                document=Document(text="", doc_id="page3"),
                error="Network timeout",
            )
        )

        # Verify event counts
        assert len(page_events) == 4  # 2 started, 1 completed, 1 skipped
        assert len(attachment_events) == 2  # 1 started, 1 processed
        assert len(error_events) == 1  # 1 page failed

        # Verify event content
        page_event_names = [event.name for event in page_events]
        assert EventName.PAGE_DATA_FETCH_STARTED in page_event_names
        assert EventName.PAGE_DATA_FETCH_COMPLETED in page_event_names
        assert EventName.PAGE_SKIPPED in page_event_names

        attachment_event_names = [event.name for event in attachment_events]
        assert EventName.ATTACHMENT_PROCESSING_STARTED in attachment_event_names
        assert EventName.ATTACHMENT_PROCESSED in attachment_event_names

        error_event_names = [event.name for event in error_events]
        assert EventName.PAGE_FAILED in error_event_names
