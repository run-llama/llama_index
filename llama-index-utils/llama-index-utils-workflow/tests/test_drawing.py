from unittest.mock import patch, MagicMock, mock_open

import pytest

from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_all_possible_flows_mermaid,
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


@pytest.mark.asyncio
async def test_draw_all_possible_flows_mermaid_basic(workflow):
    """Test basic Mermaid diagram generation."""
    with patch("builtins.open", mock_open()) as mock_file:
        result = draw_all_possible_flows_mermaid(
            workflow, filename="test_mermaid.mermaid"
        )

        # Verify file was written
        mock_file.assert_called_once_with("test_mermaid.mermaid", "w")

        # Verify basic structure
        assert isinstance(result, str)
        assert result.startswith("flowchart TD")

        # Verify contains style definitions
        assert "classDef stepStyle fill:#ADD8E6" in result
        assert "classDef startEventStyle fill:#E27AFF" in result
        assert "classDef stopEventStyle fill:#FFA07A" in result
        assert "classDef defaultEventStyle fill:#90EE90" in result
        assert "classDef externalStyle fill:#BEDAE4" in result


@pytest.mark.asyncio
async def test_draw_all_possible_flows_mermaid_no_file(workflow):
    """Test Mermaid diagram generation without file output."""
    result = draw_all_possible_flows_mermaid(workflow, filename=None)

    # Should still return the diagram string
    assert isinstance(result, str)
    assert result.startswith("flowchart TD")


@pytest.mark.asyncio
async def test_mermaid_node_shapes_and_styles(workflow):
    """Test that Mermaid nodes have correct shapes and styles."""
    result = draw_all_possible_flows_mermaid(workflow, filename=None)

    lines = result.split("\n")

    # Check for step nodes (should use box shape [...] and stepStyle)
    step_nodes = [line for line in lines if "step_" in line and ":::stepStyle" in line]
    for step_line in step_nodes:
        assert "[" in step_line and "]" in step_line, (
            f"Step node should use box shape: {step_line}"
        )
        assert ":::stepStyle" in step_line, (
            f"Step node should use stepStyle: {step_line}"
        )

    # Check for event nodes (should use ellipse shape ([...]) and event styles)
    event_nodes = [
        line
        for line in lines
        if "event_" in line
        and (
            ":::startEventStyle" in line
            or ":::stopEventStyle" in line
            or ":::defaultEventStyle" in line
        )
    ]
    for event_line in event_nodes:
        assert "([" in event_line and "])" in event_line, (
            f"Event node should use ellipse shape: {event_line}"
        )


@pytest.mark.asyncio
async def test_mermaid_edges_generation(workflow):
    """Test that Mermaid edges are properly generated."""
    result = draw_all_possible_flows_mermaid(workflow, filename=None)

    lines = result.split("\n")
    edge_lines = [line.strip() for line in lines if " --> " in line]

    # Should have at least some edges
    assert len(edge_lines) > 0, "Should generate at least some edges"

    # All edge lines should follow the pattern: source --> target
    for edge_line in edge_lines:
        assert edge_line.count(" --> ") == 1, (
            f"Edge should have exactly one arrow: {edge_line}"
        )
        source, target = edge_line.split(" --> ")
        assert source.strip(), f"Edge source should not be empty: {edge_line}"
        assert target.strip(), f"Edge target should not be empty: {edge_line}"


@pytest.mark.asyncio
async def test_mermaid_id_cleaning(workflow):
    """Test that Mermaid IDs are properly cleaned for validity."""
    result = draw_all_possible_flows_mermaid(workflow, filename=None)

    lines = result.split("\n")

    # Check that all node IDs are valid (no spaces, special chars that would break Mermaid)
    for line in lines:
        if line.strip().startswith(("step_", "event_", "external_step")):
            # Extract the ID (first word)
            parts = line.strip().split()
            if parts:
                node_id = parts[0]
                # Should not contain spaces, dots, or hyphens
                assert " " not in node_id, (
                    f"Node ID should not contain spaces: {node_id}"
                )
                assert "." not in node_id, f"Node ID should not contain dots: {node_id}"
                # Note: We allow underscores as they're valid in Mermaid


@pytest.mark.asyncio
async def test_mermaid_vs_pyvis_consistency(workflow):
    """Test that Mermaid and Pyvis generate consistent node/edge counts."""
    # Generate Pyvis version
    with patch("pyvis.network.Network") as mock_network:
        mock_net_instance = MagicMock()
        mock_network.return_value = mock_net_instance

        draw_all_possible_flows(workflow, filename="test.html")

        # Count unique nodes (Pyvis deduplicates automatically)
        pyvis_unique_nodes = set()
        for call in mock_net_instance.add_node.call_args_list:
            args, kwargs = call
            node_id = args[0]  # First argument is the node ID
            pyvis_unique_nodes.add(node_id)

        pyvis_edge_calls = len(mock_net_instance.add_edge.call_args_list)

    # Generate Mermaid version
    mermaid_result = draw_all_possible_flows_mermaid(workflow, filename=None)
    lines = mermaid_result.split("\n")

    # Count Mermaid nodes (lines with node definitions, but NOT edge lines)
    mermaid_node_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Node lines start with step_, event_, or external_step BUT don't contain -->
        if " --> " not in line and line.startswith(
            ("step_", "event_", "external_step")
        ):
            mermaid_node_lines.append(line)

    # Count Mermaid edges (lines with arrows)
    mermaid_edge_lines = [line for line in lines if " --> " in line]

    # Should have same number of nodes and edges
    assert len(mermaid_node_lines) == len(pyvis_unique_nodes), (
        f"Mermaid nodes ({len(mermaid_node_lines)}) should match Pyvis unique nodes ({len(pyvis_unique_nodes)})"
    )
    assert len(mermaid_edge_lines) == pyvis_edge_calls, (
        f"Mermaid edges ({len(mermaid_edge_lines)}) should match Pyvis edges ({pyvis_edge_calls})"
    )


@pytest.mark.asyncio
async def test_mermaid_file_writing(workflow):
    """Test that Mermaid diagram is correctly written to file."""
    mock_file_handle = mock_open()

    with patch("builtins.open", mock_file_handle):
        result = draw_all_possible_flows_mermaid(
            workflow, filename="test_output.mermaid"
        )

        # Verify file was opened for writing
        mock_file_handle.assert_called_once_with("test_output.mermaid", "w")

        # Verify content was written
        written_content = "".join(
            call.args[0] for call in mock_file_handle().write.call_args_list
        )

        assert written_content == result, "File content should match returned string"
        assert written_content.startswith("flowchart TD"), (
            "File should contain valid Mermaid syntax"
        )


@pytest.mark.asyncio
async def test_mermaid_empty_filename(workflow):
    """Test that Mermaid works with empty/None filename."""
    # Test with None
    result1 = draw_all_possible_flows_mermaid(workflow, filename=None)
    assert isinstance(result1, str)
    assert result1.startswith("flowchart TD")

    # Test with empty string
    result2 = draw_all_possible_flows_mermaid(workflow, filename="")
    assert isinstance(result2, str)
    assert result2.startswith("flowchart TD")

    # Both should be identical
    assert result1 == result2
