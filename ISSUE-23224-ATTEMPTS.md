# Issue #23224 attempt log

Repository commit: 2232f3019f7aa581f455583262100111dff4a4d3 (`main`, version 0.14.25)
Environment: Windows, Python 3.13.14; editable `llama-index-core` installed.

## Baseline

- Reproduction command: inline Python using `HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])` over 200 numbered sentences.
- Result: 20 non-root nodes; 15 have PREVIOUS; 0 have NEXT. Of 16 leaves, 11 original-document slices using start_char_idx/end_char_idx do not equal leaf text.
- Status: confirmed both reported defects.

## Approaches tried

1. `git clone` full repository: stalled while fetching; abandoned before source changes.
2. Shallow filtered clone: fetched Git pack successfully but checkout initially ran asynchronously and left a lock during concurrent inspection. Checkout later completed at the target commit. No source files were changed by these attempts.
3. Install editable `llama-index-core`: succeeded and enabled local reproduction.

## Push-blocker resolution

- First push failed because the feature branch was based on upstream `main`, while the user's fork had diverged. GitHub evaluated the entire comparison against the fork default branch, which included changes to `.github/workflows/*.yml`, requiring the token's `workflow` scope.
- Resolution: base a clean feature branch on the fork's current `main`, cherry-pick only the issue fix, resolve the changelog conflict using the fork's version plus the single issue entry, and inspect the fork-to-branch diff before pushing.
- Successful branch: `fix/issue-23224-clean`. Only the changelog, issue-specific attempt log, parser implementation, and regression test are changed. Push succeeded without changing token scopes.
- Also encountered a stale `.git/shallow.lock` with no active Git process; removing that specific stale lock allowed the fetch to proceed.
