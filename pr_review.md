# Code Review: morongosteve/ouroboros

**Repository:** morongosteve/ouroboros (fork of joi-lab/ouroboros → razzant/ouroboros)
**Reviewed:** 2026-02-25
**Version:** 6.2.0
**Language:** Python

---

## Executive Summary

Ouroboros is a self-modifying AI agent that reads and rewrites its own source code via git, governed by a philosophical constitution (`BIBLE.md`). The codebase is ambitious and technically interesting, but carries significant security and correctness issues that must be addressed before any production adoption. Three open PRs (#5, #6, #7) on the upstream `joi-lab/ouroboros` repo each have issues requiring attention.

**Critical issues found: 5 | Medium issues: 6 | Low/quality issues: 9**

---

## Architecture Overview

```
Telegram  ──►  colab_launcher.py
                      │
               supervisor/          (process management layer)
               ├── events.py        event dispatch → events.jsonl
               ├── git_ops.py       git operations
               ├── queue.py         task queue + scheduling
               ├── state.py         state + budget tracking (Google Drive)
               ├── telegram.py      Telegram client
               └── workers.py       worker lifecycle
                      │
               ouroboros/           (agent core)
               ├── agent.py         thin orchestrator (OuroborosAgent)
               ├── consciousness.py background daemon: wakes, thinks, messages owner
               ├── context.py       LLM context + prompt caching
               ├── llm.py           OpenRouter client (LLMClient)
               ├── loop.py          tool loop + concurrent execution (~980 lines)
               ├── memory.py        scratchpad, identity, chat
               └── tools/           plugin registry (auto-discovery)
                   ├── core.py      file ops
                   ├── git.py       git ops
                   ├── github.py    GitHub Issues
                   ├── shell.py     shell + Claude Code CLI
                   ├── search.py    web search
                   ├── control.py   restart, evolve, review
                   ├── browser.py   Playwright (stealth)
                   └── review.py    multi-model review
```

Data flow: `Telegram → supervisor → task queue → OuroborosAgent.handle_task() → run_llm_loop() → LLMClient.chat() → OpenRouter API → tool execution → events.jsonl / Drive`

The agent receives commands via Telegram, decomposes them into tasks, executes tool loops via LLM, and can commit changes back to its own git repository — thereby self-modifying. State is persisted on Google Drive across Colab restarts.

---

## Security Analysis

### Critical: `apply_patch.py` Installs a System Binary at Import Time

`apply_patch.py`'s `install()` function is called as a **side effect during module initialization**. On import, it silently writes an executable to `/usr/local/bin/apply_patch` (or, in PR #5/#6's variant, a hardcoded user-specific path). This is a supply-chain attack vector.

More critically, the patch parser has **no path traversal protection**. It calls `pathlib.Path(path)` and `p.parent.mkdir(parents=True, exist_ok=True)` on paths extracted from patch content without any sandboxing or allowed-path list. A maliciously crafted patch can write arbitrary files anywhere on the filesystem, e.g.:

```
*** Add File: /etc/cron.d/backdoor
```

**Recommendation:** Never write executables as a side effect of import. Add an explicit allowlist of paths the patcher may write to (e.g., only within the repo working directory). Never call `mkdir(parents=True)` on LLM-derived paths.

---

### Critical: Auto-Commit and Auto-Push on Startup Without Confirmation

In `agent.py → _check_uncommitted_changes()`, if dirty files are found on startup, the code automatically runs `git add -u && git commit && git push` with no owner confirmation. The branch-name regex check (`^[a-zA-Z0-9_/-]+$`) provides minimal protection — it does not prevent a misconfigured env var from targeting `origin/main` or another protected branch.

Additionally, on push failure the code runs `git reset HEAD~1`. If the process is interrupted between push-failure detection and the reset, a commit is destroyed without recovery.

**Recommendation:** Require explicit owner confirmation (Telegram reply) before any auto-push. Never use `git reset HEAD~1` as a recovery path; use `git push --force-with-lease` with proper conflict handling instead.

---

### Critical: Unrestricted Shell Execution

`tools/shell.py` exposes a shell execution tool to the LLM with no sandbox, no command allowlist, and no human approval gate. Combined with the self-modification loop, a single compromised LLM response can:
- Exfiltrate all API keys from the environment
- Install malicious packages via `pip install`
- Execute `curl | bash` payloads from attacker-controlled URLs

The BIBLE.md further **explicitly grants** the agent permission to solve captchas, automate browsers, and work with any website or API — including with Playwright stealth mode to bypass bot detection. There are no domain restrictions or rate limits on browser automation.

**Recommendation:** Implement a shell command allowlist or blocklist of high-risk patterns. Consider requiring Telegram confirmation for any command not matching a safe pattern.

---

### Critical: Direct Injection of Owner Messages into LLM Stream

In `loop.py → _drain_incoming_messages()`, raw owner message text is injected directly as `{"role": "user", "content": injected}` into the live LLM message stream with no sanitization or privilege distinction from the system prompt. A compromised Telegram account, or a MITM on the Google Drive mailbox file (`owner_inject.py`), can inject arbitrary LLM instructions mid-task, bypassing whatever system-prompt boundaries exist.

**Recommendation:** Mark injected owner messages with a distinct prefix in the system prompt that instructs the LLM to treat them as operational updates, not new commands. Validate that injected content does not contain prompt-injection patterns (e.g., "Ignore all previous instructions").

---

### Critical: `_get_pricing()` Double-Checked Locking Race

In `loop.py`, the pricing initialization has a broken double-checked locking pattern:

```python
_pricing_fetched = True   # ← set BEFORE the fetch attempt
try:
    ...fetch live pricing...
except Exception:
    _pricing_fetched = False  # ← reset on failure
```

If two threads race before the fetch completes, the second thread reads `_pricing_fetched = True` on the fast path and returns the partially-initialized `_cached_pricing = dict(_MODEL_PRICING_STATIC)` — live pricing data is never merged. Budget calculations will silently use stale static prices.

**Recommendation:** Use a `threading.Lock` to guard the initialization, or initialize pricing eagerly at startup outside the hot path.

---

### High: Budget Enforcement Is Self-Reported and Leaky

`supervisor/state.py` budget tracking relies on the agent correctly reporting its own token usage. v6.2.0 patched "double budget accounting" — a prior bug shows this accounting can be wrong. Additionally, `_check_budget_limits()` in `loop.py` checks `budget_pct > 0.5` using `accumulated_cost / budget_remaining_usd`. If `budget_remaining_usd` is very small (e.g., $0.01) and a single LLM call costs $0.03, `budget_pct = 3.0` triggers a hard-stop — which makes *another* LLM call to compose the stopping message, potentially spending beyond the remaining budget.

**Recommendation:** Enforce budget externally via the LLM provider's billing limits. Use a circuit breaker that halts immediately without an LLM call when remaining budget falls below one estimated call cost.

---

### High: Telegram Authentication — No Owner Verification

The bot accepts commands from any user who can message it. There is no documented `OWNER_CHAT_ID` filter. If the bot token leaks, anyone can send `/evolve`, `/restart`, or arbitrary LLM messages to fully control the agent.

**Recommendation:** Hard-code or env-configure an `OWNER_CHAT_ID` and reject all messages from other chat IDs before any processing.

---

### Medium: Advisory File Lock Race in `state.py`

`state.py → acquire_file_lock()` uses `O_CREAT | O_EXCL` (advisory lock) with stale-lock detection. Two processes can both read `st_mtime`, both decide the lock is stale (age > stale_sec), and both attempt `lock_path.unlink()`. The winner proceeds; the loser silently skips. The result is silent lock loss under contention.

**Recommendation:** Use `fcntl.flock()` or a proper advisory lock library. Alternatively, write the PID into the lock file and verify the lock-holder PID is still alive before breaking the lock.

---

### Medium: Google Drive State Has No Integrity Check

State and mailbox files on Google Drive have no signing or HMAC. External modification of these files (by anyone with Drive folder access) would go undetected and could alter agent behavior on restart.

**Recommendation:** Sign state files with an HMAC key derived from a secret (e.g., a hash of the GitHub token).

---

### Low: GitHub Token Scope

`GITHUB_TOKEN` is required and used for committing self-modifications. Personal access tokens typically have write access to all user repositories. A compromised agent could write to unintended repositories.

**Recommendation:** Use a fine-grained GitHub token scoped only to the specific fork.

---

## Code Quality Findings

### `loop.py` Violates the Project's Own BIBLE.md

`loop.py` is ~980 lines. BIBLE.md Principle 5 (Minimalism) states: "a module fits in one context window, ~1000 lines." The file mixes LLM orchestration, tool dispatch, budget guards, retry logic, context compaction, self-check injection, dynamic tool wiring, browser executor lifecycle, and per-task Drive mailbox draining — far too many responsibilities.

**Recommendation:** Extract tool dispatch and retry logic into `ouroboros/dispatch.py`. Extract budget guard and self-check injection into `ouroboros/guards.py`.

---

### Pricing Table Duplicated Across Files

`_MODEL_PRICING_STATIC` appears identically in both `loop.py` (lines 29–45) and in `llm.py`'s pricing logic. There is no single source of truth. When a new model is added, it must be updated in two places.

**Recommendation:** Centralize model pricing in a single `ouroboros/pricing.py` module imported by both.

---

### `consciousness.py` Creates an Independent ToolRegistry

`consciousness.py → _build_registry()` creates a second `ToolRegistry` instance, independent from the main agent's registry. Tools registered dynamically after boot in the main agent are invisible to the consciousness loop. The two registries share the same underlying Drive paths but maintain separate in-memory state — a latent consistency bug.

**Recommendation:** Pass the agent's existing registry to the consciousness loop rather than constructing a new one.

---

### `_verify_restart()` Claim File Can Leak

`agent.py → _verify_restart()` writes a PID-named claim file, then reads it back, then unlinks it. If the process crashes between write and unlink, the claim file leaks permanently. There is no orphan-cleanup logic on startup.

**Recommendation:** Add a startup sweep that removes claim files older than a reasonable threshold (e.g., 5 minutes) or whose PID is no longer running.

---

### `status_text()` Does O(n) Disk Scan on Every Call

`budget_breakdown()` and `model_breakdown()` in `state.py` open and scan `events.jsonl` linearly on every call. `status_text()` calls both in sequence. In a long-running session with megabytes of events, this is a blocking linear scan on every `/status` request. Only `per_task_cost_summary()` implements the `tail_bytes` optimization.

**Recommendation:** Apply the `tail_bytes` optimization to `budget_breakdown()` and `model_breakdown()`, or maintain running totals in the in-memory state object.

---

### Unpinned Dependencies

`requirements.txt` contains only 4 lines with no version pins:

```
openai
requests
playwright
playwright-stealth
```

This means `pip install` produces a non-reproducible environment. A new `openai` SDK major version could silently break the `resp.model_dump()` call pattern in `llm.py`.

**Recommendation:** Pin all dependencies with exact versions (`openai==1.x.y`) and add a `requirements-dev.txt` for development-only packages. Use `pip-compile` or Poetry to manage pins.

---

### Background Consciousness Concurrency Risk

`consciousness.py → _execute_tool()` sets `self._registry._ctx.current_chat_id` and `pending_events` directly without a lock, before every tool call. If the consciousness thread races with the agent thread — possible since consciousness `resume()` is called after task completion, not before tool execution completes — these shared fields can be corrupted.

**Recommendation:** All mutations to shared context fields must go through a lock or be replaced with thread-local state.

---

## Open PRs Review

### PR #5 — "fix: install apply_patch to user-local bin on linux server"
**Branch:** `takahacomore:ouroboros-stable` → `joi-lab:main`
**Files changed:** 3 | **+43 / -10**

**Recommendation: Request changes**

1. **Critical:** `apply_patch.py` — replaces `/usr/local/bin/apply_patch` with `/home/alexroll/.local/bin/apply_patch`, hard-coding the `alexroll` username. This will break on any other system. Correct fix:
   ```python
   APPLY_PATCH_PATH = pathlib.Path(
       os.environ.get("APPLY_PATCH_PATH",
           str(pathlib.Path.home() / ".local" / "bin" / "apply_patch"))
   )
   ```
2. **Positive:** The Colab guard (`try/except ImportError` around `google.colab`) is a good improvement that allows local development without Colab.
3. **Positive:** Supporting `MISTRAL_API_KEY` as an alternative to `OPENROUTER_API_KEY` widens accessibility.
4. **Positive:** Making `DRIVE_ROOT` and `REPO_DIR` configurable and context-aware is the right direction.
5. **Minor:** `LLM_API_KEY = OPENROUTER_API_KEY or MISTRAL_API_KEY` — the precedence is implicit. Document that OpenRouter takes priority over Mistral.
6. **Neutral:** This PR and PR #6 are near-identical duplicates. Only one should be merged; close the other.

---

### PR #6 — "fix: use user-local apply_patch path on linux"
**Branch:** `takahacomore:ouroboros` → `joi-lab:main`
**Files changed:** 3 | **+43 / -10**

**Recommendation: Close as duplicate of PR #5**

Functionally identical to PR #5. PR #6 has `mergeable_state: clean` while PR #5 has `mergeable_state: unknown`. If one is to be merged after the `alexroll` hardcoding is fixed, prefer #6 (clean merge state) and close #5.

---

### PR #7 — "Codex/OpenAI direct api"
**Branch:** `oleynik-alina:codex/openai-direct-api` → `joi-lab:main`
**Files changed:** 78 | **+5,792 / -236**

**Recommendation: Block — do not merge**

1. **No PR description.** A 78-file, ~6K-line PR with zero description is not reviewable. The author must explain: what problem does this solve, how does the OpenAI direct API differ from OpenRouter, and what is the `viktor-friday` skills framework?
2. **Scope is too large.** Adding an entirely new skills/plugin framework (`.claude/skills/` tree with `manifest.yaml` schemas, `scripts/vfriday_skill_apply.py`) and an OpenAI direct API pathway in the same PR is unacceptable for review. Split into separate PRs.
3. **Security concern:** The `manifest.yaml`-driven skill system supports `post_apply` and `test` shell commands. This is arbitrary shell execution triggered by skill installation — it expands the agent's attack surface significantly and needs dedicated security review before any merge.
4. **`add-lean4-verifier` skill** — adds a Lean4 formal verification scaffold with no documented integration path into the existing multi-model review pipeline.

---

## Summary Table

| # | Finding | Severity |
|---|---------|----------|
| 1 | `apply_patch` path traversal + silent install at import | Critical |
| 2 | Auto-commit/push on startup without confirmation | Critical |
| 3 | Unrestricted shell execution + captcha/browser automation | Critical |
| 4 | Raw owner message injection into LLM stream | Critical |
| 5 | `_get_pricing()` double-checked locking race | Critical |
| 6 | Budget enforcement is self-reported and leaky | High |
| 7 | No Telegram owner authentication | High |
| 8 | Advisory file lock race in `state.py` | Medium |
| 9 | Drive state has no integrity check | Medium |
| 10 | GitHub token scope too broad | Low |
| 11 | `loop.py` ~980 lines, too many responsibilities | Medium |
| 12 | Pricing table duplicated in `loop.py` and `llm.py` | Low |
| 13 | Consciousness loop creates independent ToolRegistry | Medium |
| 14 | Claim file leaks on crash in `_verify_restart()` | Low |
| 15 | `status_text()` does O(n) disk scan per call | Medium |
| 16 | No version pins in `requirements.txt` | Medium |
| 17 | Consciousness/agent thread share mutable context fields | Medium |
| 18 | PR #5/#6: username hardcoded in apply_patch path | Critical (for that PR) |
| 19 | PR #7: no description, too large, `post_apply` shell risk | Blocker |

---

## Overall Assessment

Ouroboros is a genuinely novel project — a self-modifying agent with a philosophical constitution and autonomous evolution. The architecture is coherent and the BIBLE.md principles are consistently applied. The v6.x series shows meaningful iteration: the deadlock fix (v6.0), selective tool schemas (v6.1), and LLM-first dedup (v6.2) all reflect real engineering progress.

However, the five critical security issues (path traversal in `apply_patch`, auto-push without confirmation, unrestricted shell + browser, direct LLM stream injection, and pricing race) must be resolved before this can be considered robust. The most urgent are the `apply_patch` installer (arbitrary file write at import time) and the LLM message injection vector (allows mid-task prompt injection via Telegram or Drive).

For a personal/experimental project run in a sandboxed Colab environment with a private Telegram bot, the current security posture may be acceptable in practice. For any wider deployment, multi-user access, or shared-repo setup, these issues are genuine risks.

**Recommended merge order for PRs:** Close #5 (duplicate), merge #6 after fixing the hardcoded username, block #7 pending description + PR split + security audit of `post_apply` execution.
