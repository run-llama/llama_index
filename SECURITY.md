# Security Policy

Before reporting a vulnerability, please review In-Scope Targets and Out-of-Scope Targets below.

## In-Scope Targets

The following packages and repositories are eligible for bug bounties:

- llama-index-core
- llama-index-integrations (see exceptions)
- llama-index-networks

## Out of Scope Targets

All out of scope targets defined by huntr as well as:

- **llama-index-experimental**: This repository is for experimental code and is not
  eligible for bug bounties, bug reports to it will be marked as interesting or waste of
  time and published with no bounty attached.
- **llama-index-integrations/tools**: Community contributed tools are not eligible for bug
  bounties. Generally tools interact with the real world. Developers are expected to
  understand the security implications of their code and are responsible for the security
  of their tools.
- Code documented with security notices. This will be decided done on a case by
  case basis, but likely will not be eligible for a bounty as the code is already
  documented with guidelines for developers that should be followed for making their
  application secure.

## Reporting LlamaCloud Vulnerabilities

Please report security vulnerabilities associated with LlamaCloud by email to `security@llamaindex.ai`.

- LlamaCloud site: https://cloud.llamaindex.ai
- LlamaCloud API: https://api.cloud.llamaindex.ai/docs
- LlamaParse client: https://github.com/run-llama/llama_parse

### Other Security Concerns

For any other security concerns, please contact us at `security@llamaindex.ai`.
