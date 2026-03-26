# LlamaIndex Readers Integration: Github

`pip install llama-index-readers-github`

The github readers package consists of three separate readers:

1. Repository Reader
2. Issues Reader
3. Collaborators Reader

## Authentication

The readers support two authentication methods:

### 1. Personal Access Token (PAT)

Generate a token under your account settings at https://github.com/settings/tokens

```python
from llama_index.readers.github import GithubClient

# Direct token
client = GithubClient(github_token="ghp_your_token_here")

# Or via environment variable
import os

os.environ["GITHUB_TOKEN"] = "ghp_your_token_here"
client = GithubClient()  # Automatically uses GITHUB_TOKEN
```

### 2. GitHub App Authentication

For better security, rate limits, and organization-level access, use GitHub App authentication:

```python
from llama_index.readers.github import GithubClient, GitHubAppAuth

# Load your GitHub App private key
with open("path/to/private-key.pem", "r") as f:
    private_key = f.read()

# Create GitHub App auth handler
app_auth = GitHubAppAuth(
    app_id="123456",  # Your GitHub App ID
    private_key=private_key,  # Private key content (PEM format)
    installation_id="789012",  # Installation ID for the target org/repo
)

# Use with any client
client = GithubClient(github_app_auth=app_auth)
```

**Installation for GitHub App support:**

```bash
pip install llama-index-readers-github[github-app]
```

**Benefits of GitHub App authentication:**

- **Higher rate limits**: 5,000 requests/hour per installation (vs 5,000/hour for PAT)
- **Fine-grained permissions**: Repository-specific access control
- **Better security**: Tokens auto-expire after 1 hour
- **Organization-level**: Can be installed across multiple repositories
- **Auditability**: Actions attributed to the app, not individual users

## Repository Reader

This reader will read through a repo, with options to specifically filter directories, file extensions, file paths, and custom processing logic.

### Basic Usage

```python
from llama_index.readers.github import GithubRepositoryReader, GithubClient

client = github_client = GithubClient(github_token=github_token, verbose=False)

reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    use_parser=False,
    verbose=True,
    filter_directories=(
        ["docs"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            "json",
            ".ipynb",
        ],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
)

documents = reader.load_data(branch="main")
```

### Advanced Filtering Options

#### Filter Specific File Paths

```python
# Include only specific files
reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    filter_file_paths=(
        ["README.md", "src/main.py", "docs/guide.md"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
)

# Exclude specific files
reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    filter_file_paths=(
        ["tests/test_file.py", "temp/cache.txt"],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
)
```

#### Custom File Processing Callback

```python
def process_file_callback(file_path: str, file_size: int) -> tuple[bool, str]:
    """Custom logic to determine if a file should be processed.

    Args:
        file_path: The full path to the file
        file_size: The size of the file in bytes

    Returns:
        Tuple of (should_process: bool, reason: str)
    """
    # Skip large files
    if file_size > 1024 * 1024:  # 1MB
        return False, f"File too large: {file_size} bytes"

    # Skip test files
    if "test" in file_path.lower():
        return False, "Skipping test files"

    # Skip binary files by extension
    binary_extensions = [".exe", ".bin", ".so", ".dylib"]
    if any(file_path.endswith(ext) for ext in binary_extensions):
        return False, "Skipping binary files"

    return True, ""


reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    process_file_callback=process_file_callback,
    fail_on_error=False,  # Continue processing if callback fails
)
```

#### Custom Folder for Temporary Files

```python
from llama_index.core.readers.base import BaseReader


# Custom parser for specific file types
class CustomMarkdownParser(BaseReader):
    def load_data(self, file_path, extra_info=None):
        # Custom parsing logic here
        pass


reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    use_parser=True,
    custom_parsers={".md": CustomMarkdownParser()},
    custom_folder="/tmp/github_processing",  # Custom temp directory
)
```

### Event System Integration

The reader integrates with LlamaIndex's instrumentation system to provide detailed events during processing:

```python
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.readers.github.repository.event import (
    GitHubFileProcessedEvent,
    GitHubFileSkippedEvent,
    GitHubFileFailedEvent,
    GitHubRepositoryProcessingStartedEvent,
    GitHubRepositoryProcessingCompletedEvent,
)


class GitHubEventHandler(BaseEventHandler):
    def handle(self, event):
        if isinstance(event, GitHubRepositoryProcessingStartedEvent):
            print(f"Started processing repository: {event.repository_name}")
        elif isinstance(event, GitHubFileProcessedEvent):
            print(
                f"Processed file: {event.file_path} ({event.file_size} bytes)"
            )
        elif isinstance(event, GitHubFileSkippedEvent):
            print(f"Skipped file: {event.file_path} - {event.reason}")
        elif isinstance(event, GitHubFileFailedEvent):
            print(f"Failed to process file: {event.file_path} - {event.error}")
        elif isinstance(event, GitHubRepositoryProcessingCompletedEvent):
            print(
                f"Completed processing. Total documents: {event.total_documents}"
            )


# Register the event handler
dispatcher = get_dispatcher()
handler = GitHubEventHandler()
dispatcher.add_event_handler(handler)

# Use the reader - events will be automatically dispatched
reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
)
documents = reader.load_data(branch="main")
```

#### Available Events

The following events are dispatched during repository processing:

- **`GitHubRepositoryProcessingStartedEvent`**: Fired when repository processing begins

  - `repository_name`: Name of the repository (owner/repo)
  - `branch_or_commit`: Branch name or commit SHA being processed

- **`GitHubRepositoryProcessingCompletedEvent`**: Fired when repository processing completes

  - `repository_name`: Name of the repository
  - `branch_or_commit`: Branch name or commit SHA
  - `total_documents`: Number of documents created

- **`GitHubTotalFilesToProcessEvent`**: Fired with the total count of files to be processed

  - `repository_name`: Name of the repository
  - `branch_or_commit`: Branch name or commit SHA
  - `total_files`: Total number of files found

- **`GitHubFileProcessingStartedEvent`**: Fired when individual file processing starts

  - `file_path`: Path to the file being processed
  - `file_type`: File extension

- **`GitHubFileProcessedEvent`**: Fired when a file is successfully processed

  - `file_path`: Path to the processed file
  - `file_type`: File extension
  - `file_size`: Size of the file in bytes
  - `document`: The created Document object

- **`GitHubFileSkippedEvent`**: Fired when a file is skipped

  - `file_path`: Path to the skipped file
  - `file_type`: File extension
  - `reason`: Reason why the file was skipped

- **`GitHubFileFailedEvent`**: Fired when file processing fails
  - `file_path`: Path to the failed file
  - `file_type`: File extension
  - `error`: Error message describing the failure

## Issues Reader

```python
from llama_index.readers.github import (
    GitHubRepositoryIssuesReader,
    GitHubIssuesClient,
)

github_client = GitHubIssuesClient(github_token=github_token, verbose=True)

reader = GitHubRepositoryIssuesReader(
    github_client=github_client,
    owner="moncho",
    repo="dry",
    verbose=True,
)

documents = reader.load_data(
    state=GitHubRepositoryIssuesReader.IssueState.ALL,
    labelFilters=[("bug", GitHubRepositoryIssuesReader.FilterType.INCLUDE)],
)
```

## Collaborators Reader

```python
from llama_index.readers.github import (
    GitHubRepositoryCollaboratorsReader,
    GitHubCollaboratorsClient,
)

github_client = GitHubCollaboratorsClient(
    github_token=github_token, verbose=True
)

reader = GitHubRepositoryCollaboratorsReader(
    github_client=github_client,
    owner="moncho",
    repo="dry",
    verbose=True,
)

documents = reader.load_data()
```

## GitHub App Setup Guide

To create and configure a GitHub App for authentication:

### 1. Create a GitHub App

1. Go to your GitHub account settings → Developer settings → GitHub Apps → **New GitHub App**
2. Fill in the required information:
   - **GitHub App name**: Choose a unique name (e.g., "My LlamaIndex Reader")
   - **Homepage URL**: Your application or organization URL
   - **Webhook**: Uncheck "Active" (not needed for this use case)

### 2. Set Permissions

Under **Repository permissions**, set:

- **Contents**: Read-only (to read repository files)
- **Metadata**: Read-only (required automatically)
- **Issues**: Read-only (if using Issues reader)
- **Pull requests**: Read-only (issues endpoint includes PRs)

### 3. Install the App

1. After creating the app, note your **App ID** (shown at the top)
2. Generate a **private key**:
   - Scroll down to "Private keys"
   - Click "Generate a private key"
   - Save the downloaded `.pem` file securely
3. Install the app:
   - Click "Install App" in the left sidebar
   - Choose the account/organization
   - Select **specific repositories** or **all repositories**
   - Complete installation

### 4. Get Installation ID

After installation, you'll be redirected to a URL like:

```
https://github.com/settings/installations/12345678
```

The number `12345678` is your **installation ID**. You can also find it via the API:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     https://api.github.com/app/installations
```

### 5. Use in Code

```python
from llama_index.readers.github import GithubClient, GitHubAppAuth

# Load private key
with open("path/to/your-app-private-key.pem", "r") as f:
    private_key = f.read()

# Create auth handler
app_auth = GitHubAppAuth(
    app_id="YOUR_APP_ID",
    private_key=private_key,
    installation_id="YOUR_INSTALLATION_ID",
)

# Use with any client
client = GithubClient(github_app_auth=app_auth)
```

### Token Management

The `GitHubAppAuth` class automatically:

- Generates JWTs for app authentication
- Obtains installation access tokens
- Caches tokens (valid for 1 hour)
- Refreshes tokens automatically when they expire or are within 5 minutes of expiry

You can manually invalidate a token if needed:

```python
app_auth.invalidate_token()  # Forces refresh on next request
```

### Troubleshooting

**"Failed to get installation token: 401"**

- Verify your App ID is correct
- Ensure the private key matches your GitHub App
- Check that the app is installed for the target repository

**"Failed to get installation token: 404"**

- Verify the installation ID is correct
- Ensure the app installation wasn't uninstalled

**"Import PyJWT failed"**

- Install GitHub App support: `pip install llama-index-readers-github[github-app]`

```

```
