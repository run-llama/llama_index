import json
from unittest.mock import MagicMock, patch

import pytest

from llama_index.tools.ilovevideoeditor import ILoveVideoEditorTool

JOB_ID = "425ba18a-dab0-4827-afc9-eec80e5b3c20"
SPEC = {
    "name": "test",
    "layers": [
        {
            "type": "text",
            "settings": {"startTime": 0, "duration": 2, "text": "hi"},
        }
    ],
}


def make_tool() -> ILoveVideoEditorTool:
    return ILoveVideoEditorTool(api_key="vf_live_test", poll_interval_seconds=0)


def mock_response(payload, status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        response.raise_for_status.return_value = None
    return response


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ILOVEVIDEOEDITOR_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ILOVEVIDEOEDITOR_API_KEY"):
        ILoveVideoEditorTool()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ILOVEVIDEOEDITOR_API_KEY", "vf_live_env")
    tool = ILoveVideoEditorTool()
    assert tool.api_key == "vf_live_env"


def test_spec_functions():
    assert ILoveVideoEditorTool.spec_functions == [
        "render_video",
        "get_render_status",
    ]


@patch("requests.request")
def test_queue_only(mock_request):
    mock_request.return_value = mock_response({"jobId": JOB_ID, "status": "queued"})
    tool = make_tool()
    result = json.loads(tool.render_video(SPEC, wait_for_completion=False))
    assert result == {"jobId": JOB_ID, "status": "queued"}
    # Verify the API key header was sent
    _, kwargs = mock_request.call_args
    assert kwargs["headers"]["x-api-key"] == "vf_live_test"


@patch("requests.request")
def test_render_to_completion(mock_request):
    mock_request.side_effect = [
        mock_response({"jobId": JOB_ID, "status": "queued"}),
        mock_response(
            {"jobId": JOB_ID, "status": "rendering", "progress": {"percent": 40}}
        ),
        mock_response(
            {
                "jobId": JOB_ID,
                "status": "completed",
                "downloadUrl": "https://cdn.example.com/video.mp4",
            }
        ),
    ]
    tool = make_tool()
    result = json.loads(tool.render_video(SPEC))
    assert result == {
        "jobId": JOB_ID,
        "status": "completed",
        "downloadUrl": "https://cdn.example.com/video.mp4",
    }


@patch("requests.request")
def test_completed_without_url_fetches_download_url(mock_request):
    mock_request.side_effect = [
        mock_response({"jobId": JOB_ID, "status": "queued"}),
        mock_response({"jobId": JOB_ID, "status": "completed"}),
        mock_response({"downloadUrl": "https://cdn.example.com/fresh.mp4"}),
    ]
    tool = make_tool()
    result = json.loads(tool.render_video(SPEC))
    assert result["downloadUrl"] == "https://cdn.example.com/fresh.mp4"


@patch("requests.request")
def test_failed_render_returns_error(mock_request):
    mock_request.side_effect = [
        mock_response({"jobId": JOB_ID, "status": "queued"}),
        mock_response({"jobId": JOB_ID, "status": "failed", "error": "invalid layer"}),
    ]
    tool = make_tool()
    result = json.loads(tool.render_video(SPEC))
    assert result["status"] == "failed"
    assert result["error"] == "invalid layer"


@patch("requests.request")
def test_timeout_returns_job_id(mock_request):
    calls = {"n": 0}

    def fake_request(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return mock_response({"jobId": JOB_ID, "status": "queued"})
        return mock_response({"jobId": JOB_ID, "status": "rendering"})

    mock_request.side_effect = fake_request
    tool = ILoveVideoEditorTool(
        api_key="vf_live_test", max_wait_seconds=0.01, poll_interval_seconds=0
    )
    result = json.loads(tool.render_video(SPEC))
    assert result["jobId"] == JOB_ID
    assert "did not finish" in result["error"]


@patch("requests.request")
def test_http_error_returns_message(mock_request):
    mock_request.return_value = mock_response({"error": "invalid api key"}, status=401)
    tool = make_tool()
    result = tool.render_video(SPEC)
    assert "Error rendering video" in result
    assert "401" in result


@patch("requests.request")
def test_get_render_status(mock_request):
    mock_request.return_value = mock_response(
        {"jobId": JOB_ID, "status": "rendering", "progress": {"percent": 55}}
    )
    tool = make_tool()
    result = json.loads(tool.get_render_status(JOB_ID))
    assert result == {"jobId": JOB_ID, "status": "rendering", "progress": 55}


@patch("requests.request")
def test_get_render_status_error(mock_request):
    mock_request.return_value = mock_response({"error": "not found"}, status=404)
    tool = make_tool()
    result = tool.get_render_status("nonexistent")
    assert "Error getting render status" in result


def test_to_tool_list():
    tool = make_tool()
    tools = tool.to_tool_list()
    assert len(tools) == 2
    names = {t.metadata.name for t in tools}
    assert names == {"render_video", "get_render_status"}
