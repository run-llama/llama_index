# New Feature

## Description

<!-- Provide a clear description of the new feature and its motivation. -->

Fixes # (issue)

## Related PRs

<!-- List any related pull requests. Remove this section if not applicable. -->

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

## Design Decisions

<!-- Describe key design decisions and any alternatives considered. -->

## API Changes

<!-- List any new or modified public APIs. Remove this section if not applicable. -->

- New classes:
- New functions:
- Modified interfaces:

## How Has This Been Tested?

Your pull-request will likely not be merged unless it is covered by some form of impactful unit testing.

Please describe the tests that you ran to verify the feature works as expected.

**Usage example:**
```python
# Show a brief usage example of the new feature
```

- [ ] I added new unit tests to cover this feature
- [ ] I added integration tests
- [ ] I believe this feature is already covered by existing unit tests

## Screenshots / Logs

<!-- If applicable, add screenshots or example output. Remove this section if not applicable. -->

## Performance Impact

<!-- Describe any performance implications of this feature. -->

- [ ] This change has no significant performance impact
- [ ] This change improves performance (please describe)
- [ ] This change may degrade performance (please describe and justify)

## Rollback Plan

- [ ] Simple revert is sufficient
- [ ] Rollback requires additional steps (please describe)

## Checklist:

- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added Google Colab support for the newly added notebooks
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I ran `uv run make format; uv run make lint` to appease the lint gods
