from unittest.mock import patch

import pytest

from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)


@pytest.mark.asyncio()
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
