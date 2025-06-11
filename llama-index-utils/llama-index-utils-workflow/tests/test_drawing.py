from unittest.mock import patch, MagicMock

import pytest

from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)


@pytest.mark.asyncio
async def test_workflow_draw_methods(workflow):
    with patch("pyvis.network.Network") as mock_network:
        draw_all_possible_flows(workflow, filename="test_all_flows.html")
        mock_network.assert_called_once()
        mock_network.return_value.show.assert_called_once_with(
            "test_all_flows.html", notebook=False
        )

    await workflow.run()
    with patch("pyvis.network.Network") as mock_network:
        draw_most_recent_execution(workflow, filename="test_recent_execution.html")
        mock_network.assert_called_once()
        mock_network.return_value.show.assert_called_once_with(
            "test_recent_execution.html", notebook=False
        )


@pytest.mark.asyncio
async def test_draw_all_possible_flows_with_max_label_length(workflow):
    """Test the max_label_length parameter."""
    with patch("pyvis.network.Network") as mock_network:
        mock_net_instance = MagicMock()
        mock_network.return_value = mock_net_instance

        # Test with max_label_length=10
        draw_all_possible_flows(
            workflow, filename="test_truncated.html", max_label_length=10
        )

        # Extract actual label mappings from add_node calls
        label_mappings = {}
        for call in mock_net_instance.add_node.call_args_list:
            _, kwargs = call
            label = kwargs.get("label")
            title = kwargs.get("title")

            # For items with titles (truncated), map title->label
            if title:
                label_mappings[title] = label
            # For items without titles (not truncated), map label->label
            elif label:
                label_mappings[label] = label

        # Test cases using actual events from DummyWorkflow fixture
        test_cases = [
            ("OneTestEvent", "OneTestEv*"),  # 12 chars -> truncated to 10
            ("LastEvent", "LastEvent"),  # 9 chars -> no truncation
            (
                "StartEvent",
                "StartEvent",
            ),  # 10 chars -> no truncation (exactly at limit)
            ("StopEvent", "StopEvent"),  # 9 chars -> no truncation
        ]

        # Verify actual results match expected for available test cases
        for original, expected_label in test_cases:
            if original in label_mappings:
                actual_label = label_mappings[original]
                assert actual_label == expected_label, (
                    f"Expected '{original}' to become '{expected_label}', but got '{actual_label}'"
                )
                assert len(actual_label) <= 10, (
                    f"Label '{actual_label}' exceeds max_label_length=10"
                )
