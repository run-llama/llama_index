# Description

Add missing `timeout` parameter to HTTP requests across 9 integration packages to prevent indefinite hangs when remote servers are unresponsive.

Fixes #22028

| Package | File | Fix |
|---------|------|-----|
| `llama-index-embeddings-huggingface` | `utils.py` | Add `timeout=30` to `requests.get()` |
| `llama-index-llms-you` | `base.py` | Add `timeout=60` to `requests.post()` (2 locations) |
| `llama-index-tools-wolfram-alpha` | `base.py` | Add `timeout=30` to `requests.get()` |
| `llama-index-tools-text-to-image` | `base.py` | Add `timeout=30` to `requests.get()` (2 locations) |
| `llama-index-tools-openapi` | `base.py` | Add `timeout=30` to `requests.get()` |
| `llama-index-readers-kaltura` | `base.py` | Add `timeout=30` to `requests.get()` |
| `llama-index-readers-zendesk` | `base.py` | Add `timeout=30` to `requests.get()` |
| `llama-index-readers-intercom` | `base.py` | Add `timeout=30` to `requests.get()` |
| `llama-index-indices-managed-dashscope` | `utils.py` | Add `timeout=60` to `requests.post/get()` (3 locations) |

## New Package?
- [x] No

## Version Bump?
- [x] No

## Type of Change
- [x] Bug fix (non-breaking change which fixes an issue)

## How Has This Been Tested?
- [x] I believe this change is already covered by existing unit tests

## Suggested Checklist:
- [x] I have performed a self-review of my own code
- [x] My changes generate no new warnings
- [x] New and existing unit tests pass locally with my changes
