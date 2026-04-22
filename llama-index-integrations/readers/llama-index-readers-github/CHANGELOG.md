# CHANGELOG

## [0.9.0]

### Added

- **GitHub App Authentication**: Added support for authenticating via GitHub App as an alternative to Personal Access Tokens (PAT)
  - New `GitHubAppAuth` class for handling GitHub App authentication with automatic token management
  - All client classes (`GithubClient`, `GitHubIssuesClient`, `GitHubCollaboratorsClient`) now support `github_app_auth` parameter
  - Automatic installation token caching with 5-minute expiry buffer (300 seconds)
  - Token auto-refresh when near expiry
  - JWT generation using RS256 algorithm for GitHub App authentication
  - Optional dependency group `[github-app]` for installing PyJWT with cryptographic support
  - Comprehensive test suite with 25 test cases covering all authentication scenarios
  - Example script demonstrating GitHub App usage patterns
  - Updated README with GitHub App setup guide and troubleshooting section

### Changed

- Client initialization now accepts either `github_token` (PAT) or `github_app_auth` (GitHub App), but not both
- `_get_auth_headers()` method is now async to support dynamic token fetching for GitHub App authentication

### Backward Compatibility

- All existing PAT-based authentication code remains fully functional and unchanged
- No breaking changes to existing APIs
- All 19 existing tests continue to pass

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
