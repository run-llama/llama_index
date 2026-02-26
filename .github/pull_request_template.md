# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Related PRs

<!-- List any related pull requests, e.g. PRs that this depends on or PRs in other repos. -->

- Related PR #

## New Package?

Did I fill in the `tool.llamahub` section in the `pyproject.toml` and provide a detailed README.md for my new integration or package?

- [ ] Yes
- [ ] No
- [ ] N/A

## Version Bump?

Did I bump the version in the `pyproject.toml` file of the package I am updating? (Except for the `llama-index-core` package)

- [ ] Yes
- [ ] No
- [ ] N/A

## Type of Change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## How Has This Been Tested?

Your pull-request will likely not be merged unless it is covered by some form of impactful unit testing.

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce.

- [ ] I added new unit tests to cover this change
- [ ] I believe this change is already covered by existing unit tests

## Screenshots / Logs

<!-- If applicable, add screenshots or log output to help explain the change. Remove this section if not applicable. -->

## Performance Impact

<!-- Describe any performance implications of this change. Remove this section if not applicable. -->

- [ ] This change has no significant performance impact
- [ ] This change improves performance (please describe)
- [ ] This change may degrade performance (please describe and justify)

## Rollback Plan

<!-- How can this change be reverted if something goes wrong? Is it a simple revert, or are there additional steps needed (e.g., database migrations, config changes)? Remove this section for trivial changes. -->

- [ ] Simple revert is sufficient
- [ ] Rollback requires additional steps (please describe)

## Suggested Checklist:

- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added Google Colab support for the newly added notebooks
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I ran `uv run make format; uv run make lint` to appease the lint gods

---

> **Tip:** Specialized templates are available for [bug fixes](?expand=1&template=bug_fix.md), [features](?expand=1&template=feature.md), [documentation](?expand=1&template=docs.md), and [breaking changes](?expand=1&template=breaking_change.md).
