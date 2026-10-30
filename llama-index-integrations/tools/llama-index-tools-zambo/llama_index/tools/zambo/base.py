"""Zambo tool spec for LlamaIndex.

Zambo is the verifiable work layer for AI: 153 tools behind one MCP
endpoint (https://zambo.dev/mcp), continuity that carries sessions,
memory, and work across AIs through HANDOVER, and a verifiable AER-1
receipt for every job, so "done" is something you can check.

This tool spec gives a LlamaIndex agent three things:

  list_tools   -- list the tools the Zambo MCP endpoint offers
  call_tool    -- call any Zambo tool by name through the MCP endpoint;
                  every call returns a verifiable receipt for the job
  verify_receipt -- verify an AER-1 workflow receipt offline, with no
                  network call and no trust in any server: the Merkle
                  root is recomputed from the receipt's own steps and
                  the required fields are checked (draft-zambo-aer1-26)

Free tier: 20 calls + 1 wrap-up free every day; picking up in another
AI is always free. The Day Pass is $0.99 for unlimited use for 24 hours.
No account and no API key are needed for the free tier.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request

from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_ENDPOINT = "https://zambo.dev/mcp"
WORKFLOW_TYPE = "verifiable-workflow-receipt"


def merkle_root(receipt_ids: list) -> str:
    """AER-1 Section 8.1 Merkle root over ordered step receipt ids.

    leaf = SHA-256 over the UTF-8 bytes of receipt_id; internal node =
    SHA-256 over raw 32-byte left || raw 32-byte right; an odd trailing
    node is duplicated.
    """
    if not receipt_ids:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(rid.encode("utf-8")).digest() for rid in receipt_ids]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def verify_workflow_receipt(workflow) -> list:
    """Offline checks for an AER-1 workflow receipt.

    Returns a list of failure strings; an empty list means the receipt
    passed every check performed here. A matching Merkle root proves
    the supplied steps agree with the supplied root. It does not prove
    the answer a job produced was right.
    """
    failures = []
    if not isinstance(workflow, dict):
        return ["workflow receipt is not a JSON object"]
    if workflow.get("type") != WORKFLOW_TYPE:
        failures.append("type is not verifiable-workflow-receipt")
    for key in ("workflow_id", "receipt_id", "merkle_root"):
        if not workflow.get(key):
            failures.append(f"missing {key}")
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        failures.append("steps is not a non-empty list")
        return failures
    seen = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            failures.append(f"step {i + 1} is not a JSON object")
            continue
        if step.get("seq") != i + 1:
            failures.append(f"step {i + 1} seq is not {i + 1}")
        rid = step.get("receipt_id")
        if not rid:
            failures.append(f"step {i + 1} missing receipt_id")
        elif rid in seen:
            failures.append(f"step {i + 1} duplicates a receipt_id")
        seen.add(rid)
        if not step.get("tool"):
            failures.append(f"step {i + 1} missing tool")
    if not failures:
        recomputed = merkle_root([s["receipt_id"] for s in steps])
        if workflow.get("merkle_root") != recomputed:
            failures.append("merkle_root does not match the recomputed root")
    return failures


class ZamboToolSpec(BaseToolSpec):
    """Tool spec wrapping the Zambo MCP endpoint and AER-1 verification."""

    spec_functions = ["list_tools", "call_tool", "verify_receipt"]

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout: float = 30.0) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout

    def _rpc(self, method: str, params: dict) -> dict:
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # MCP endpoints may answer with a plain JSON body or an SSE frame.
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line.startswith("{"):
                return json.loads(line)
        return json.loads(raw)

    def list_tools(self) -> str:
        """List the tools offered by the Zambo MCP endpoint.

        Returns a JSON string with the tool count and the tool names.
        Zambo offers 153 tools in total, 36 on the default surface.
        """
        try:
            result = self._rpc("tools/list", {})
        except Exception as exc:  # network or endpoint problem: say so plainly
            return f"ERROR: could not reach the Zambo MCP endpoint: {exc}"
        tools = (result.get("result") or {}).get("tools") or []
        names = [t.get("name", "") for t in tools if isinstance(t, dict)]
        return json.dumps({"count": len(names), "tools": names})

    def call_tool(self, tool_name: str, arguments_json: str = "{}") -> str:
        """Call a Zambo tool by name through the MCP endpoint.

        Args:
            tool_name (str): Name of the Zambo tool to call, for example
                "handover" to wrap up work so another AI can pick it up.
            arguments_json (str): The tool arguments as a JSON object
                string. Defaults to an empty object.

        Every call returns a verifiable receipt for the job. Free tier:
        20 calls + 1 wrap-up free every day; picking up in another AI
        is always free.
        """
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
        except ValueError as exc:
            return f"ERROR: arguments_json is not valid JSON: {exc}"
        if not isinstance(arguments, dict):
            return "ERROR: arguments_json must be a JSON object"
        try:
            result = self._rpc(
                "tools/call", {"name": tool_name, "arguments": arguments}
            )
        except Exception as exc:
            return f"ERROR: could not reach the Zambo MCP endpoint: {exc}"
        body = result.get("result") or {}
        parts = []
        for item in body.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        if parts:
            return "\n".join(parts)
        return json.dumps(body if body else result)

    def verify_receipt(self, receipt_json: str) -> str:
        """Verify an AER-1 workflow receipt offline.

        Args:
            receipt_json (str): The workflow receipt as a JSON string,
                as emitted by a Zambo call or by an AER-1 collector.

        The Merkle root is recomputed from the receipt's own steps and
        the required fields are checked locally. No network call is
        made and no server is trusted. Returns "verified" with a short
        summary when the receipt passes, or "verification failed" with
        the reasons when it does not.
        """
        try:
            workflow = json.loads(receipt_json)
        except (ValueError, TypeError) as exc:
            return f"verification failed: receipt is not valid JSON: {exc}"
        failures = verify_workflow_receipt(workflow)
        if failures:
            return "verification failed: " + "; ".join(failures)
        return (
            "verified: AER-1 workflow receipt "
            f"{workflow.get('receipt_id')} with "
            f"{len(workflow.get('steps') or [])} step(s); merkle_root "
            "matches the root recomputed from the ordered steps"
        )
