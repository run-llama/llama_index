# Stale Package Policy

## Overview

To maintain high quality across all LlamaIndex integrations, we periodically review packages for staleness. Packages marked as "stale" are moved to a `stale_packages` branch but remain published on PyPI. This policy ensures that our main branch contains only well-maintained, tested integrations while preserving access to historical integrations.

## Staleness Criteria

Packages are evaluated using an automated health check that considers:

1. **Test Coverage (50% of score)**

   - Full credit (1.0): 5+ test functions
   - Partial credit (0.5): 2-4 test functions
   - No credit (0.0): 0-1 test functions

2. **Download Activity (40% of score)**

   - Measured relative to llama-index-core
   - Considers monthly download counts
   - Weighted towards recent activity

3. **Commit Activity (10% of score)**
   - Measured relative to llama-index-core
   - Considers commit frequency and consistency
   - Weighted towards recent commits

The resulting score is then used to determine the health of the package.

## Moving to Stale Status

A package may be moved to the `stale_packages` branch if:

1. It has a low health score (typically below 0.005)
2. AND lacks adequate test coverage
3. OR has known breaking issues that haven't been addressed

The final decision to mark a package as stale involves human review and is not purely automated.

## Effects of Stale Status

When a package is marked as stale:

1. The package code is moved to the `stale_packages` branch
2. Documentation is removed from the main documentation site
3. The package remains published on PyPI

## Reactivating a Stale Package

Any contributor can reactivate a stale package by:

1. Creating a PR to move any package from the `stale_packages` branch to `main`
2. Ensuring the package has adequate test coverage (minimum 2 tests)
3. Verifying that all tests pass
4. Updating documentation as needed

## Questions or Concerns

If you maintain a package that has been marked as stale, or have questions about this policy:

1. Open a GitHub issue for discussion
2. Reach out to the maintainers on Discord
3. Submit a PR to reactivate your package with improvements

We aim to be transparent and collaborative in maintaining package quality while preserving access to all integrations.
