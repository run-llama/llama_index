"""
Taskmarket tool spec.

Taskmarket (https://taskmarket.dev) is an onchain agent labor marketplace on
Base. This spec lets a LlamaIndex agent discover open tasks, inspect task
detail and submission status, and (with explicit authorization and a spend
cap) create a funded task through the official ``taskmarket`` CLI.

Reads go straight to the public Taskmarket REST API and need no wallet.
The funded write path shells out to the first-party CLI so that wallet keys,
X402 payment, legal acceptance, and idempotency are handled by official
tooling -- this spec never touches private keys or tokens.
"""

import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

TASKMARKET_API_BASE = "https://api.taskmarket.dev/api"
DEFAULT_MAX_SPEND = float(os.environ.get("TASKMARKET_MAX_SPEND", "5.0"))


def _get_json(url: str, timeout: int = 45) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


class TaskmarketToolSpec(BaseToolSpec):
    """
    Taskmarket tool spec.

    Exposes discovery (``list_tasks``), status tracking (``get_task``),
    submission review (``list_submissions``), and a spend-capped, explicitly
    authorized task creation flow (``create_task``).
    """

    spec_functions = [
        "list_tasks",
        "get_task",
        "list_submissions",
        "create_task",
    ]

    def __init__(
        self,
        api_base: str = TASKMARKET_API_BASE,
        max_spend: float = DEFAULT_MAX_SPEND,
        cli_path: str = "taskmarket",
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.max_spend = max_spend
        self.cli_path = cli_path

    def list_tasks(
        self,
        status: str = "open",
        phase: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 25,
    ) -> str:
        """
        List Taskmarket tasks (default: open tasks, newest first).

        Args:
            status (str): Filter by status, e.g. "open" (default).
            phase (str): Optional phase filter: active, in_review,
                awaiting_settlement, resolved.
            mode (str): Optional mode filter: bounty, claim, pitch, benchmark,
                auction.
            limit (int): Maximum number of tasks to return (default 25).

        """
        params = {"status": status, "take": str(min(max(limit, 1), 100))}
        if phase:
            params["phase"] = phase
        if mode:
            params["mode"] = mode
        qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        data = _get_json(f"{self.api_base}/tasks?{qs}")
        tasks = data.get("tasks", [])
        rows = []
        for t in tasks[:limit]:
            reward = t.get("reward", "0")
            try:
                usdc = round(int(reward) / 1_000_000, 6)
            except (TypeError, ValueError):
                usdc = reward
            rows.append(
                {
                    "id": t.get("id"),
                    "title": (t.get("description") or "")[:120],
                    "rewardUSDC": usdc,
                    "status": t.get("status"),
                    "phase": t.get("phase"),
                    "submissionWindowOpen": t.get("submissionWindowOpen"),
                    "expiryTime": t.get("expiryTime"),
                }
            )
        return json.dumps(rows, indent=2)

    def get_task(self, task_id: str) -> str:
        """
        Get the full live status of one Taskmarket task.

        Args:
            task_id (str): The full 64-hex Taskmarket task id
                (e.g. 0x...). Fetch it from list_tasks if unsure.

        """
        data = _get_json(f"{self.api_base}/tasks/{task_id}")
        reward = data.get("reward", "0")
        try:
            usdc = round(int(reward) / 1_000_000, 6)
        except (TypeError, ValueError):
            usdc = reward
        out = {
            "id": data.get("id"),
            "status": data.get("status"),
            "phase": data.get("phase"),
            "mode": data.get("mode"),
            "rewardUSDC": usdc,
            "awardCount": data.get("awardCount"),
            "submissionCount": data.get("submissionCount"),
            "submissionWindowOpen": data.get("submissionWindowOpen"),
            "expiryTime": data.get("expiryTime"),
        }
        return json.dumps(out, indent=2)

    def list_submissions(self, task_id: str) -> str:
        """
        List submissions for a Taskmarket task for human review.

        This is a read-only action: it NEVER accepts or rejects any
        submission. A human requester decides via the official tooling.

        Args:
            task_id (str): The full 64-hex Taskmarket task id.

        """
        data = _get_json(f"{self.api_base}/tasks/{task_id}/submissions")
        subs = data if isinstance(data, list) else data.get("submissions", [])
        rows = []
        for s in subs:
            rows.append(
                {
                    "id": s.get("id"),
                    "workerAddress": s.get("workerAddress"),
                    "workerAgentId": s.get("workerAgentId"),
                    "submittedAt": s.get("submittedAt"),
                    "rejectedAt": s.get("rejectedAt"),
                    "deliverableHash": s.get("deliverableHash"),
                    "submitTxHash": s.get("submitTxHash"),
                    "fileUrl": s.get("fileUrl"),
                }
            )
        return json.dumps(rows, indent=2)

    def create_task(
        self,
        description: str,
        reward: float,
        duration_hours: int,
        mode: str = "bounty",
        authorization: Optional[str] = None,
        task_visibility: str = "public",
    ) -> str:
        """
        Create a funded Taskmarket task through the official CLI.

        SAFETY CONTRACT (enforced here, before any money moves):
        - The exact cost (reward + platform fees, in USDC on Base) is
          computed and SURFACED below; if it exceeds ``self.max_spend`` the
          call refuses before invoking anything.
        - ``authorization`` must be a fresh, explicit string supplied by the
          caller (e.g. an operator or a separate approval step) confirming
          the exact amount. No authorization string -> no payment.
        - The actual transfer is delegated to the first-party ``taskmarket``
          CLI (wallet keys, X402 payment, legal acceptance, and idempotency
          are the CLI's responsibility). This tool never stores or logs
          private keys, seeds, or tokens.
        - Result handling polls task status by id; it never blindly retries
          a payment whose settlement status is unknown.

        Args:
            description (str): Full task description with deliverables and
                acceptance criteria.
            reward (float): Reward in USDC (e.g. 5.0 for 5 USDC).
            duration_hours (int): Task duration in hours.
            mode (str): Task mode: bounty (default), claim, pitch, benchmark,
                auction.
            authorization (str): Fresh explicit authorization string. Must
                include the exact reward amount, e.g. "authorize 5 USDC for
                taskmarket task".
            task_visibility (str): public (default), unlisted, or private.

        """
        if not authorization:
            return (
                "REFUSED: no authorization. create_task requires a fresh, "
                "explicit authorization string confirming the exact reward "
                f'amount (e.g. "authorize {reward} USDC for taskmarket '
                'task"). Nothing was created.'
            )
        if authorization.strip() != f"authorize {reward} USDC for taskmarket task":
            return (
                "REFUSED: authorization string must exactly match "
                f'"authorize {reward} USDC for taskmarket task". '
                "Nothing was created."
            )
        if reward <= 0:
            return "REFUSED: reward must be > 0 USDC. Nothing was created."
        if reward > self.max_spend:
            return (
                f"REFUSED: reward {reward} USDC exceeds max_spend "
                f"{self.max_spend} USDC (set TASKMARKET_MAX_SPEND to raise). "
                "Nothing was created."
            )
        if duration_hours <= 0:
            return "REFUSED: duration_hours must be > 0. Nothing was created."
        if not shutil.which(self.cli_path):
            return (
                "REFUSED: 'taskmarket' CLI not found on PATH. Install it via "
                "npm (official package) to create funded tasks. Nothing was "
                "created."
            )

        cmd = [
            self.cli_path,
            "task",
            "create",
            "--description",
            description,
            "--reward",
            str(reward),
            "--duration",
            str(duration_hours),
            "--mode",
            mode,
            "--task-visibility",
            task_visibility,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return (
                "taskmarket task create failed (exit "
                f"{proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )

        out = proc.stdout.strip()
        task_id = self._extract_task_id(out)
        return json.dumps(
            {
                "created": True,
                "taskId": task_id,
                "taskUrl": f"https://taskmarket.dev/tasks/{task_id}"
                if task_id
                else None,
                "cliOutput": out,
                "liveStatus": task_id and json.loads(self.get_task(task_id)),
            },
            indent=2,
        )

    @staticmethod
    def _extract_task_id(cli_output: str) -> Optional[str]:
        """Pull the 64-hex task id out of CLI output, if present."""
        for token in cli_output.replace("\n", " ").split():
            t = token.strip(".,:;'\"")
            if t.startswith("0x") and len(t) == 66:
                return t
        return None
