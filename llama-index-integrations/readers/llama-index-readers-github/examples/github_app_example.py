"""
Example: Using GitHub App Authentication with LlamaIndex GitHub Reader

This example demonstrates how to use GitHub App authentication instead of a Personal Access Token (PAT).
GitHub App authentication is more secure and provides better rate limits for enterprise use cases.

Prerequisites:
1. Create a GitHub App (see README.md for detailed setup guide)
2. Install the GitHub App in your organization/account
3. Download the private key (.pem file)
4. Note your App ID and Installation ID

Installation:
    pip install llama-index-readers-github[github-app]
"""

import os
from pathlib import Path
from llama_index.readers.github import GithubRepositoryReader, GitHubAppAuth


def read_private_key_from_file(key_path: str) -> str:
    """Read private key from a file."""
    with open(key_path, "r") as f:
        return f.read()


def example_with_github_app():
    """Example: Load repository using GitHub App authentication."""

    # Step 1: Set up GitHub App credentials
    # You can get these values from your GitHub App settings
    app_id = os.getenv("GITHUB_APP_ID", "123456")
    installation_id = os.getenv("GITHUB_INSTALLATION_ID", "789012")

    # Load private key from file or environment variable
    private_key_path = os.getenv("GITHUB_PRIVATE_KEY_PATH", "path/to/your-app.private-key.pem")

    if Path(private_key_path).exists():
        private_key = read_private_key_from_file(private_key_path)
    else:
        # Alternatively, you can store the key in an environment variable
        private_key = os.getenv("GITHUB_PRIVATE_KEY", "")

    if not private_key:
        print("Error: GitHub App private key not found!")
        print("Please set GITHUB_PRIVATE_KEY_PATH or GITHUB_PRIVATE_KEY environment variable")
        return

    # Step 2: Create GitHubAppAuth instance
    github_app_auth = GitHubAppAuth(
        app_id=app_id,
        private_key=private_key,
        installation_id=installation_id,
    )

    # Step 3: Create reader with GitHub App authentication
    reader = GithubRepositoryReader(
        owner="facebook",
        repo="react",
        github_app_auth=github_app_auth,  # Use GitHub App auth instead of github_token
        verbose=True,
    )

    # Step 4: Load documents
    # The reader will automatically fetch and refresh installation tokens as needed
    print("Loading repository...")
    documents = reader.load_data(branch="main")

    print(f"Loaded {len(documents)} documents from the repository")

    # Example: Print first document
    if documents:
        print(f"\nFirst document preview:")
        print(f"File: {documents[0].metadata.get('file_path', 'N/A')}")
        print(f"Content length: {len(documents[0].text)} characters")


def example_with_filtering():
    """Example: Load specific files with GitHub App authentication."""

    # Set up credentials
    app_id = os.getenv("GITHUB_APP_ID")
    installation_id = os.getenv("GITHUB_INSTALLATION_ID")
    private_key = os.getenv("GITHUB_PRIVATE_KEY")

    if not all([app_id, installation_id, private_key]):
        print("Missing GitHub App credentials!")
        return

    # Create auth and reader
    github_app_auth = GitHubAppAuth(
        app_id=app_id,
        private_key=private_key,
        installation_id=installation_id,
    )

    reader = GithubRepositoryReader(
        owner="python",
        repo="cpython",
        github_app_auth=github_app_auth,
        verbose=True,
        # Filter to only include Python files in the Lib directory
        filter_file_extensions=(
            [".py"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    )

    print("Loading Python files from Lib directory...")
    documents = reader.load_data(branch="main")

    print(f"Loaded {len(documents)} Python files")


def example_token_management():
    """Example: Manual token management with GitHub App."""

    app_id = os.getenv("GITHUB_APP_ID")
    installation_id = os.getenv("GITHUB_INSTALLATION_ID")
    private_key = os.getenv("GITHUB_PRIVATE_KEY")

    if not all([app_id, installation_id, private_key]):
        print("Missing GitHub App credentials!")
        return

    github_app_auth = GitHubAppAuth(
        app_id=app_id,
        private_key=private_key,
        installation_id=installation_id,
    )

    # The token is automatically fetched and cached
    print("First token fetch (will make API call)...")
    token1 = github_app_auth.get_installation_token()
    print(f"Token expires at: {github_app_auth._token_expires_at}")

    # Subsequent calls use cached token
    print("\nSecond fetch (uses cache)...")
    token2 = github_app_auth.get_installation_token()
    assert token1 == token2, "Token should be cached"
    print("Token was retrieved from cache")

    # Force refresh the token
    print("\nForce refresh...")
    token3 = github_app_auth.get_installation_token(force_refresh=True)
    print(f"New token fetched, expires at: {github_app_auth._token_expires_at}")

    # Manual invalidation (useful if you know the token was revoked)
    print("\nInvalidating token manually...")
    github_app_auth.invalidate_token()
    print("Token cache cleared")

    # Next call will fetch a fresh token
    token4 = github_app_auth.get_installation_token()
    print("Fresh token fetched after invalidation")


if __name__ == "__main__":
    print("=" * 60)
    print("GitHub App Authentication Examples")
    print("=" * 60)

    # Choose which example to run
    example = os.getenv("EXAMPLE", "basic")

    if example == "basic":
        print("\nRunning basic example...")
        example_with_github_app()
    elif example == "filtering":
        print("\nRunning filtering example...")
        example_with_filtering()
    elif example == "token_mgmt":
        print("\nRunning token management example...")
        example_token_management()
    else:
        print(f"\nUnknown example: {example}")
        print("Valid options: basic, filtering, token_mgmt")
        print("Set EXAMPLE environment variable to choose")
