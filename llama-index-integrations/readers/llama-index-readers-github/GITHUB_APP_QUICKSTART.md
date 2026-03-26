# GitHub App Authentication - Quick Start Guide

This guide will help you quickly set up and use GitHub App authentication with the llama-index-readers-github package.

## Prerequisites

1. **Install the package with GitHub App support:**

   ```bash
   pip install llama-index-readers-github[github-app]
   ```

2. **Create a GitHub App** (one-time setup):

   - Go to GitHub Settings → Developer settings → GitHub Apps → New GitHub App
   - Fill in required fields (name, homepage URL, webhook URL - can use placeholder)
   - Set permissions:
     - Repository permissions → Contents: Read-only
     - (Optional) Issues: Read-only (if using issues reader)
   - Click "Create GitHub App"
   - **Note your App ID** from the app settings page

3. **Generate a private key:**

   - On your GitHub App settings page, scroll to "Private keys"
   - Click "Generate a private key"
   - Save the downloaded `.pem` file securely

4. **Install the GitHub App:**
   - Go to your GitHub App settings → Install App
   - Choose which account/organization to install it on
   - Select repositories (all or specific ones)
   - **Note your Installation ID** from the URL: `https://github.com/settings/installations/{installation_id}`

## Quick Start Code

```python
from llama_index.readers.github import GithubRepositoryReader, GitHubAppAuth

# 1. Set up authentication
github_app_auth = GitHubAppAuth(
    app_id="123456",  # Your GitHub App ID
    private_key=open("your-app.pem").read(),  # Your private key
    installation_id="789012",  # Your installation ID
)

# 2. Create reader
reader = GithubRepositoryReader(
    owner="owner-name",
    repo="repo-name",
    github_app_auth=github_app_auth,
)

# 3. Load documents
documents = reader.load_data(branch="main")
print(f"Loaded {len(documents)} documents")
```

## Environment Variables Method

For better security, use environment variables:

```python
import os
from llama_index.readers.github import GithubRepositoryReader, GitHubAppAuth

# Load from environment variables
github_app_auth = GitHubAppAuth(
    app_id=os.environ["GITHUB_APP_ID"],
    private_key=os.environ["GITHUB_PRIVATE_KEY"],
    installation_id=os.environ["GITHUB_INSTALLATION_ID"],
)

reader = GithubRepositoryReader(
    owner="owner-name",
    repo="repo-name",
    github_app_auth=github_app_auth,
)

documents = reader.load_data(branch="main")
```

Set environment variables in your shell:

```bash
export GITHUB_APP_ID="123456"
export GITHUB_INSTALLATION_ID="789012"
export GITHUB_PRIVATE_KEY="$(cat your-app.private-key.pem)"
```

## Key Features

### Automatic Token Management

Tokens are automatically:

- ✅ Fetched when first needed
- ✅ Cached in memory
- ✅ Refreshed before expiration (5-minute buffer)
- ✅ Handled transparently - you don't need to worry about it!

### Error Handling

```python
from llama_index.readers.github import (
    GitHubAppAuth,
    GitHubAppAuthenticationError,
)

try:
    github_app_auth = GitHubAppAuth(
        app_id="123456",
        private_key="invalid-key",
        installation_id="789012",
    )
    token = github_app_auth.get_installation_token()
except GitHubAppAuthenticationError as e:
    print(f"Authentication failed: {e}")
```

### Force Token Refresh

```python
# Manually refresh the token if needed
github_app_auth.get_installation_token(force_refresh=True)

# Invalidate cached token (useful if token is revoked)
github_app_auth.invalidate_token()
```

### GitHub Enterprise Support

```python
github_app_auth = GitHubAppAuth(
    app_id="123456",
    private_key="...",
    installation_id="789012",
    github_base_url="https://github.your-company.com/api/v3",  # Your GHE URL
)
```

## Common Issues

### Issue: `ImportError: cannot import name 'GitHubAppAuth'`

**Solution:** Install with GitHub App support:

```bash
pip install llama-index-readers-github[github-app]
```

### Issue: `GitHubAppAuthenticationError: Failed to generate JWT`

**Solution:** Check that your private key is valid:

- Should have RSA headers
- Should be the complete key including header and footer
- Should be the `.pem` file downloaded from GitHub

### Issue: `401 Unauthorized` when fetching token

**Possible causes:**

- Invalid App ID
- Invalid private key
- Invalid Installation ID
- App not installed in the target organization/account

**Solution:** Verify your credentials:

1. Check App ID in GitHub App settings
2. Regenerate and download a new private key
3. Reinstall the app and note the new Installation ID

### Issue: `403 Forbidden` when accessing repository

**Possible causes:**

- GitHub App doesn't have access to the repository
- Insufficient permissions

**Solution:**

1. Go to GitHub Settings → Installations
2. Click "Configure" on your app
3. Add the repository to the app's access list
4. Ensure "Contents" permission is set to "Read"

## Comparison: PAT vs GitHub App

| Feature        | Personal Access Token (PAT) | GitHub App                           |
| -------------- | --------------------------- | ------------------------------------ |
| **Rate Limit** | 5,000 requests/hour         | 5,000 requests/hour per installation |
| **Scope**      | User-level access           | Repository/organization-level        |
| **Security**   | Broader access              | Granular permissions                 |
| **Expiration** | Manual rotation             | Automatic (1-hour tokens)            |
| **Audit Logs** | Limited                     | Detailed                             |
| **Best For**   | Personal projects           | Team/enterprise use                  |

## Next Steps

- See `examples/github_app_example.py` for more detailed examples
- Read the full README.md for advanced features
- Check IMPLEMENTATION_SUMMARY.md for technical details

## Support

If you encounter issues:

1. Check the troubleshooting section in README.md
2. Verify your GitHub App setup
3. Review error messages carefully
4. Open an issue on the llama-index repository with details

---

**Pro Tip:** Store your private key securely (e.g., using a secrets manager or environment variables). Never commit `.pem` files to version control!
