import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Direct import to allow standalone testing without full workspace virtualenv
_action_gate_file = (
    Path(__file__).resolve().parents[2]
    / "llama_index"
    / "core"
    / "security"
    / "action_gate.py"
)
spec = importlib.util.spec_from_file_location("action_gate", str(_action_gate_file))
action_gate = importlib.util.module_from_spec(spec)
sys.modules["action_gate"] = action_gate
spec.loader.exec_module(action_gate)

ActionBoundary = action_gate.ActionBoundary
ActionGate = action_gate.ActionGate
ActionLedger = action_gate.ActionLedger
Disposition = action_gate.Disposition
ToolTier = action_gate.ToolTier


class TestActionGate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_file = Path(self.temp_dir) / "action_ledger.jsonl"
        self.ledger = ActionLedger(self.ledger_file)
        self.gate = ActionGate(ledger=self.ledger, prove_token=None)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if "AAG_KILL_SWITCH" in os.environ:
            del os.environ["AAG_KILL_SWITCH"]

    def test_read_tool_allowed(self):
        decision = self.gate.evaluate("vector_search_documents", {"query": "Q3 Revenue"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.ALLOW)
        self.assertEqual(decision.tier, ToolTier.READ)
        self.assertFalse(decision.simulation_mode)
        self.assertTrue(len(decision.receipt_hash) == 64)

    def test_mutating_write_simulation_fallback(self):
        decision = self.gate.evaluate("insert_sql_record", {"table": "invoices", "amount": 500})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.SIMULATE)
        self.assertEqual(decision.tier, ToolTier.WRITE_MUTATING)
        self.assertTrue(decision.simulation_mode)

    def test_mutating_write_authorized_with_token(self):
        gate_with_token = ActionGate(ledger=self.ledger, prove_token="valid-prove-token")
        decision = gate_with_token.evaluate("insert_sql_record", {"table": "invoices", "amount": 500})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.ALLOW)
        self.assertFalse(decision.simulation_mode)

    def test_destructive_tool_denied_without_token(self):
        decision = self.gate.evaluate("drop_sql_table", {"table": "audit_logs"})
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.DENY)
        self.assertEqual(decision.tier, ToolTier.DESTRUCTIVE)

    def test_destructive_tool_authorized_with_token(self):
        gate_with_token = ActionGate(ledger=self.ledger, prove_token="admin-token-777")
        decision = gate_with_token.evaluate("drop_sql_table", {"table": "temp_staging"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.ALLOW)

    def test_kill_switch_blocks_all(self):
        os.environ["AAG_KILL_SWITCH"] = "1"
        decision = self.gate.evaluate("query_index", {"q": "test"})
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.disposition, Disposition.DENY)
        self.assertIn("Kill-switch engaged", decision.reason)

    def test_hash_chain_integrity(self):
        self.gate.evaluate("retrieve_nodes", {"k": 5})
        self.gate.evaluate("update_document", {"doc_id": "doc123"})
        self.gate.evaluate("truncate_table", {"table": "cache"})

        lines = self.ledger_file.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 3)

        prev_hash = "0" * 64
        for line in lines:
            record = json.loads(line)
            self.assertEqual(record["prev_hash"], prev_hash)

            canonical_args = json.dumps(record["arguments"], sort_keys=True, default=str)
            expected_payload = (
                f"{prev_hash}|{record['timestamp']}|{record['tool_name']}|{canonical_args}|"
                f"{record['decision']['disposition']}|{record.get('agent_id') or ''}"
            )
            expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
            self.assertEqual(record["receipt_hash"], expected_hash)
            prev_hash = record["receipt_hash"]

    def test_action_boundary_decorator_allow(self):
        boundary = ActionBoundary(self.gate)

        @boundary.guard
        def search_docs(query: str):
            return f"Found docs for {query}"

        result = search_docs(query="compliance")
        self.assertEqual(result, "Found docs for compliance")

    def test_action_boundary_decorator_simulate(self):
        boundary = ActionBoundary(self.gate)

        @boundary.guard
        def insert_data(row: dict):
            return f"Row {row} inserted"

        result = insert_data(row={"id": 1})
        self.assertIsInstance(result, dict)
        self.assertTrue(result["simulation_mode"])
        self.assertEqual(result["disposition"], "simulate")

    def test_action_boundary_decorator_deny(self):
        boundary = ActionBoundary(self.gate)

        @boundary.guard
        def drop_table(table: str):
            return f"Table {table} dropped"

        with self.assertRaises(PermissionError) as ctx:
            drop_table(table="users")
        self.assertIn("ActionGate blocked tool", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
