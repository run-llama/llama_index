# Description

Fix resource leaks across 5 integration packages where file handles and HTTP connections were not properly closed, preventing file descriptor exhaustion in long-running processes.

Fixes #22026

| Package | File | Issue | Fix |
|---------|------|-------|-----|
| `llama-index-readers-huggingface-fs` | `base.py` | `gzip.open()` without close | Wrapped in `with` context manager |
| `llama-index-readers-alibabacloud-aisearch` | `base.py` | `open(file_path, "rb").read()` without close (2 locations) | Wrapped in `with` context manager |
| `llama-index-readers-stripe-docs` | `base.py` | `urllib.request.urlopen()` without close | Wrapped in `with` context manager |
| `llama-index-readers-sec-filings` | `section.py` | `gzip.open()` without close | Wrapped in `with` context manager |
| `llama-index-retrievers-galaxia` | `base.py` | `HTTPSConnection` never closed | Added `try/finally` with `conn.close()` |

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
