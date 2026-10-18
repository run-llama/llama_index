import json
from unittest.mock import MagicMock, patch
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.extra_desktop import ExtraDesktopToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in ExtraDesktopToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    spec = ExtraDesktopToolSpec()
    expected = [
        "launch_app",
        "click",
        "type_text",
        "send_hotkey",
        "capture_screen",
        "snap_layout",
        "batch_actions",
        "run_subtask",
    ]
    assert spec.spec_functions == expected


@patch("extra.fastpath.shell.launch_app")
def test_launch_app(mock_launch):
    mock_launch.return_value = {"pid": 5555}
    spec = ExtraDesktopToolSpec()
    res = spec.launch_app("calc")
    data = json.loads(res)
    assert data["status"] == "launched"
    assert data["app"] == "calc"
    mock_launch.assert_called_once_with("calc", args=[])


@patch("extra.core.input_engine.instant_type")
def test_type_text(mock_type):
    mock_type.return_value = True
    spec = ExtraDesktopToolSpec()
    res = spec.type_text("100+200=", press_enter=True)
    data = json.loads(res)
    assert data["status"] == "typed"
    assert data["length"] == 8
    assert data["press_enter"] is True
    mock_type.assert_called_once_with("100+200=", press_enter=True)


@patch("extra.core.capture.capture_screen")
def test_capture_screen(mock_capture):
    mock_capture.return_value = "/tmp/llama_screen.png"
    spec = ExtraDesktopToolSpec()
    res = spec.capture_screen()
    data = json.loads(res)
    assert data["status"] == "captured"
    assert "llama_screen.png" in data["screenshot_path"]
