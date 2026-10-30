"""Offline tests for llama-index-tools-zambo.

No network: verification paths are local, and the MCP paths are pointed
at an unreachable local endpoint to check the graceful error string.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.tools.zambo import ZamboToolSpec, merkle_root, verify_workflow_receipt


def _receipt():
    steps = [
        {"seq": 1, "receipt_id": "11111111-1111-4111-8111-111111111111", "tool": "search"},
        {"seq": 2, "receipt_id": "22222222-2222-4222-8222-222222222222", "tool": "summarize"},
    ]
    return {
        "type": "verifiable-workflow-receipt",
        "workflow_id": "33333333-3333-4333-8333-333333333333",
        "receipt_id": "44444444-4444-4444-8444-444444444444",
        "merkle_root": merkle_root([s["receipt_id"] for s in steps]),
        "steps": steps,
    }


def test_tool_list_shape():
    spec = ZamboToolSpec()
    tools = spec.to_tool_list()
    names = sorted(t.metadata.name for t in tools)
    assert names == ["call_tool", "list_tools", "verify_receipt"]


def test_verify_good_receipt():
    spec = ZamboToolSpec()
    out = spec.verify_receipt(json.dumps(_receipt()))
    assert out.startswith("verified:"), out


def test_verify_changed_receipt_fails():
    spec = ZamboToolSpec()
    receipt = _receipt()
    receipt["steps"][0]["tool"] = ""
    out = spec.verify_receipt(json.dumps(receipt))
    assert out.startswith("verification failed"), out
    receipt2 = _receipt()
    receipt2["merkle_root"] = "0" * 64
    assert verify_workflow_receipt(receipt2) != []


def test_verify_bad_json_fails_cleanly():
    spec = ZamboToolSpec()
    assert spec.verify_receipt("not json").startswith("verification failed")


def test_mcp_unreachable_returns_error_string():
    spec = ZamboToolSpec(endpoint="http://127.0.0.1:9/mcp", timeout=2.0)
    assert spec.list_tools().startswith("ERROR:")
    assert spec.call_tool("handover", "{}").startswith("ERROR:")


def test_call_tool_bad_arguments():
    spec = ZamboToolSpec()
    assert spec.call_tool("handover", "not json").startswith("ERROR:")
