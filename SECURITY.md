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

## Reporting LlamaCloud Vulnerabilities

Please report security vulnerabilities associated with LlamaCloud by email to `security@llamaindex.ai`.

- LlamaCloud site: https://cloud.llamaindex.ai
- LlamaCloud API: https://api.cloud.llamaindex.ai/docs
- LlamaParse client: https://github.com/run-llama/llama_parse

### Other Security Concerns

For any other security concerns, please contact us at `security@llamaindex.ai`.
