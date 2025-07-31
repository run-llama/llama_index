# LlamaIndex Readers Integration: Github

`pip install llama-index-readers-github`

The github readers package consists of three separate readers:

1. Repository Reader
2. Issues Reader
3. Collaborators Reader

All three readers will require a personal access token (which you can generate under your account settings).

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
