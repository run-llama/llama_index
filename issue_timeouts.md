# Title
Missing timeout on HTTP requests across 9 integration packages

# Bug Description
Several integration packages make HTTP requests using `requests.get()` or `requests.post()` without specifying a `timeout` parameter. This means requests can hang indefinitely if the remote server is unresponsive, causing the calling process to freeze.

# Affected Packages

| Package | File | Line(s) | Call |
|---------|------|---------|------|
| `llama-index-embeddings-huggingface` | `utils.py` | 85 | `requests.get(pooling_config_url)` |
| `llama-index-llms-you` | `base.py` | 25, 35 | `requests.post(base_url, ...)` (2 locations) |
| `llama-index-tools-wolfram-alpha` | `base.py` | 81 | `requests.get(url, headers=headers)` |
| `llama-index-tools-text-to-image` | `base.py` | 55, 74 | `requests.get(url)` (2 locations) |
| `llama-index-tools-openapi` | `base.py` | 35 | `requests.get(url)` |
| `llama-index-readers-kaltura` | `base.py` | 177 | `requests.get(cap_json_url)` |
| `llama-index-readers-zendesk` | `base.py` | 86 | `requests.get(url)` |
| `llama-index-readers-intercom` | `base.py` | 90 | `requests.get(url, headers=headers)` |
| `llama-index-indices-managed-dashscope` | `utils.py` | 6, 16, 37 | `requests.post/get()` (3 locations) |

# Expected Behavior
All HTTP requests should include a reasonable `timeout` parameter to prevent indefinite hangs.

# Steps to Reproduce
1. Use any of the affected packages in a production environment
2. If the remote API server becomes unresponsive or has network issues
3. The `requests.get()`/`requests.post()` call hangs indefinitely with no timeout
4. The calling thread/process freezes and cannot recover

# Version
v0.14.22
