# Breaking Change

## Description

<!-- Provide a clear description of the breaking change and why it is necessary. -->

Fixes # (issue)

## Related PRs

<!-- List any related pull requests. Remove this section if not applicable. -->

- Related PR #

## Version Bump?

Did I bump the version in the `pyproject.toml` file of the package I am updating? (Except for the `llama-index-core` package)

- [ ] Yes
- [ ] No

## What Breaks

<!-- Clearly describe what existing functionality will stop working. -->

**Affected APIs / interfaces:**
-

**Previous behavior:**

**New behavior:**

## Migration Guide

<!-- Provide clear instructions for users to migrate from the old behavior to the new one. -->

### Before

```python
# Old usage
```

### After

```python
# New usage
```

## Why This Is Necessary

<!-- Explain why this breaking change is justified and what alternatives were considered. -->

## How Has This Been Tested?

Your pull-request will likely not be merged unless it is covered by some form of impactful unit testing.

- [ ] I added new unit tests to cover this change
- [ ] I updated existing tests to match the new behavior
- [ ] I have verified that deprecated code paths are properly handled

## Performance Impact

- [ ] This change has no significant performance impact
- [ ] This change improves performance (please describe)
- [ ] This change may degrade performance (please describe and justify)

## Rollback Plan

<!-- Breaking changes need a clear rollback strategy. -->

- [ ] Simple revert is sufficient
- [ ] Rollback requires additional steps (please describe)

## Checklist:

- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have updated the migration guide / changelog
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my change works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I ran `uv run make format; uv run make lint` to appease the lint gods
