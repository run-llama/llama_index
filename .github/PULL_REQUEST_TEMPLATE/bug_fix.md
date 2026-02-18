# Bug Fix

## Issue

Fixes # (issue)

## Root Cause

<!-- Describe the root cause of the bug. -->

## Fix Description

<!-- Describe the fix and why this approach was chosen. -->

## Related PRs

<!-- List any related pull requests. Remove this section if not applicable. -->

- Related PR #

## Version Bump?

Did I bump the version in the `pyproject.toml` file of the package I am updating? (Except for the `llama-index-core` package)

- [ ] Yes
- [ ] No
- [ ] N/A

## How Has This Been Tested?

Your pull-request will likely not be merged unless it is covered by some form of impactful unit testing.

Please describe the tests that you ran to verify the fix. Include steps to reproduce the original bug.

**Steps to reproduce the bug:**
1.
2.
3.

**Expected behavior:**

**Actual behavior (before fix):**

- [ ] I added new unit tests to cover this fix
- [ ] I believe this fix is already covered by existing unit tests

## Screenshots / Logs

<!-- If applicable, add before/after screenshots or log output. Remove this section if not applicable. -->

## Performance Impact

- [ ] This change has no significant performance impact
- [ ] This change improves performance (please describe)
- [ ] This change may degrade performance (please describe and justify)

## Rollback Plan

- [ ] Simple revert is sufficient
- [ ] Rollback requires additional steps (please describe)

## Checklist:

- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have verified the fix resolves the original issue
- [ ] I have checked for regressions in related functionality
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective
- [ ] New and existing unit tests pass locally with my changes
- [ ] I ran `uv run make format; uv run make lint` to appease the lint gods
