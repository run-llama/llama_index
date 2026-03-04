# Security Policy

Before reporting a vulnerability, please review In-Scope Targets and Out-of-Scope Targets below.

## In-Scope Targets

Only the main umbrella package `llama-index` is eligible for bug bounties through the
[Huntr platform](https://huntr.com/repos/run-llama/llama_index), along with its first-degree dependencies part of this
Github repository:

- llama-index-agent-openai
- llama-index-cli
- llama-index-core
- llama-index-embeddings-openai
- llama-index-indices-managed-llama-cloud
- llama-index-llms-openai
- llama-index-multi-modal-llms-openai
- llama-index-program-openai
- llama-index-question-gen-openai
- llama-index-readers-file
- llama-index-readers-llama-parse

## Out of Scope Targets

All out of scope targets defined by huntr as well as:

- Code documented with security notices. This will be decided done on a case by
  case basis, but likely will not be eligible for a bounty as the code is already
  documented with guidelines for developers that should be followed for making their
  application secure.
- Any integration hosted in this GitHub repository that's not a first-degree dependency
  of the `llama-index` package.

Out of scope targets can still be responsibly disclosed to the team through the
["Report a vulnerability"](https://github.com/run-llama/llama_index/security/advisories/new) button on GitHub.

## Threat Model

`llama-index` is a Python library intended to be used inside a trusted execution environment (e.g., server-side backend code, scheduled jobs, local scripts). When you embed `llama-index` in a network-facing service (FastAPI, Flask, Django, etc.) you are creating a new attack surface that is **outside of the scope** of this project.

Specifically:

- Validation and sanitization of **all** user-supplied input (including text, URLs, file paths, and model parameters) is the responsibility of the hosting application.
- Authentication, authorization, rate limiting and other common Web security controls are similarly out of scope.
- `llama-index` assumes that the caller has already performed any necessary filtering, size limiting, and content-type validation before data reaches the library.

## Out-of-Scope Vulnerability Classes

The following classes of issues will **not** be accepted as security vulnerabilities in `llama-index` because they can only be exploited through the web layer that wraps the library:

- **Server-Side Request Forgery (SSRF)** or open redirect arising from passing attacker-controlled URLs to helpers such as `resolve_image`, `download_loader`, or any loader/reader component.
- **Deserialization, path traversal, or file-system** read/write primitives that require the application to pass untrusted file paths or file-like objects into the library.
- **Prompt-injection or prompt-leak** style attacks that rely on manipulating the natural-language prompts provided to the library. These are inherent to the LLM threat landscape and must be mitigated at the application layer.
- **Denial of Service** caused by unbounded or malformed user input. While algorithmic complexity issues inside the library are in scope, crashes or memory exhaustion that require sending arbitrarily large payloads to an exposed API endpoint are out of scope.
- Any **OWASP Top-10** style vulnerability (e.g., XSS, SQLi, CSRF) that occurs in the developer-provided HTTP handler code rather than inside `llama-index` itself.

If you are uncertain whether a finding is in scope please open a discussion before submitting a report.

## Examples of In-Scope Vulnerability Reports

Below are hypothetical findings that would qualify for a security bounty because they exploit a weakness **inside `llama-index` itself** and can be reproduced with a minimal Python script (no web framework required):

1. **Race Condition in Temporary File Handling**
   A predictable filename in `/tmp` is opened using `open(..., 'w')` without `O_EXCL`, enabling an attacker to pre-create a symlink to another file and cause data corruption or privilege escalation.

2. **Sensitive Data Exposure via Debug Logging**
   Enabling the library's built-in debug mode causes full request/response bodies—including secrets such as `OPENAI_API_KEY`—to be written to world-readable log files by default.

3. **Insecure Temporary File Handling**
   Intermediate artifacts are written to a world-readable path in `/tmp` with predictable filenames, enabling local attackers to replace or read sensitive data.

4. **Disabled TLS Verification in Outbound Requests**
   A helper that fetches external resources sets `verify=False` for the `requests` call, allowing man-in-the-middle attacks against every user of the helper.

These examples are illustrative, not exhaustive. If you discover a different issue that compromises confidentiality, integrity, or availability _within the library_, please report it.

## Reporting LlamaCloud Vulnerabilities

Please report security vulnerabilities associated with LlamaCloud by email to `security@llamaindex.ai`.

- LlamaCloud site: https://cloud.llamaindex.ai
- LlamaCloud API: https://api.cloud.llamaindex.ai/docs
- LlamaParse client: https://github.com/run-llama/llama_parse

### Other Security Concerns

For any other security concerns, please contact us at `security@llamaindex.ai`.
