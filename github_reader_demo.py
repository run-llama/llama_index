#!/usr/bin/env python3
"""
Comprehensive demonstration of llama-index-readers-github functionality.

This script demonstrates:
1. Repository Reader - Reading source code and files from GitHub repos
2. Issues Reader - Reading GitHub issues and converting them to documents  
3. Collaborators Reader - Reading repository collaborator information
4. Advanced filtering and configuration options
5. Event handling for progress tracking

Requirements:
- Install: pip install llama-index-readers-github
- Set GITHUB_TOKEN environment variable with your GitHub personal access token
"""

import os
import sys
from typing import List
from pathlib import Path

# Add the github reader to path (for development purposes)
sys.path.insert(0, str(Path(__file__).parent / "llama-index-integrations/readers/llama-index-readers-github"))

try:
    from llama_index.readers.github import (
        # Original readers
        GithubRepositoryReader, 
        GithubClient,
        GitHubRepositoryIssuesReader, 
        GitHubIssuesClient,
        GitHubRepositoryCollaboratorsReader, 
        GitHubCollaboratorsClient,
        # NEW readers we just added
        GitHubRepositoryCommitsReader,
        GitHubCommitsClient,
        GitHubRepositoryPullRequestsReader,
        GitHubPullRequestsClient,
    )
    from llama_index.core.schema import Document
    from llama_index.core.instrumentation import get_dispatcher
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler
    from llama_index.readers.github.repository.event import (
        GitHubFileProcessedEvent,
        GitHubFileSkippedEvent,
        GitHubFileFailedEvent,
        GitHubRepositoryProcessingStartedEvent,
        GitHubRepositoryProcessingCompletedEvent,
        GitHubTotalFilesToProcessEvent,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure llama-index-readers-github is installed:")
    print("pip install llama-index-readers-github")
    sys.exit(1)


class GitHubEventHandler(BaseEventHandler):
    """Event handler to track GitHub reader progress."""
    
    def handle(self, event):
        if isinstance(event, GitHubRepositoryProcessingStartedEvent):
            print(f"🚀 Started processing repository: {event.repository_name}")
            print(f"   Branch/Commit: {event.branch_or_commit}")
        elif isinstance(event, GitHubTotalFilesToProcessEvent):
            print(f"📊 Total files to process: {event.total_files}")
        elif isinstance(event, GitHubFileProcessedEvent):
            print(f"✅ Processed file: {event.file_path} ({event.file_size} bytes)")
        elif isinstance(event, GitHubFileSkippedEvent):
            print(f"⏭️  Skipped file: {event.file_path} - {event.reason}")
        elif isinstance(event, GitHubFileFailedEvent):
            print(f"❌ Failed to process file: {event.file_path} - {event.error}")
        elif isinstance(event, GitHubRepositoryProcessingCompletedEvent):
            print(f"🎉 Completed processing. Total documents: {event.total_documents}")
            print("-" * 60)


def setup_event_handling():
    """Set up event handling for progress tracking."""
    dispatcher = get_dispatcher()
    handler = GitHubEventHandler()
    dispatcher.add_event_handler(handler)
    print("✅ Event handling set up successfully\n")


def get_github_token() -> str:
    """Get GitHub token from environment variables."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN environment variable not set!")
        print("Please create a GitHub personal access token and set it:")
        print("export GITHUB_TOKEN='your_token_here'")
        print("\nTo create a token:")
        print("1. Go to GitHub Settings > Developer settings > Personal access tokens")
        print("2. Generate new token with 'repo' permissions")
        sys.exit(1)
    return token


def demo_repository_reader_basic(github_token: str):
    """Demonstrate basic repository reading functionality."""
    print("=" * 60)
    print("🔍 DEMO 1: Basic Repository Reader")
    print("=" * 60)
    
    # Create GitHub client
    github_client = GithubClient(github_token=github_token, verbose=True)
    
    # Create repository reader with basic configuration
    reader = GithubRepositoryReader(
        github_client=github_client,
        owner="toluwajibodu",  # Your actual repository
        repo="agentic-data-gen",
        use_parser=False,
        verbose=True,
    )
    
    try:
        # Load documents from main branch
        print("📖 Loading documents from main branch...")
        documents = reader.load_data(branch="main")
        
        print(f"\n📊 Results:")
        print(f"   Total documents created: {len(documents)}")
        
        # Show first few documents
        for i, doc in enumerate(documents[:3]):
            print(f"\n📄 Document {i+1}:")
            print(f"   ID: {doc.doc_id}")
            print(f"   Content preview: {doc.text[:200]}...")
            if doc.extra_info:
                print(f"   Extra info: {doc.extra_info}")
                
    except Exception as e:
        print(f"❌ Error reading repository: {e}")


def demo_repository_reader_advanced(github_token: str):
    """Demonstrate advanced repository reading with filtering."""
    print("\n" + "=" * 60)
    print("🔍 DEMO 2: Advanced Repository Reader with Filtering")
    print("=" * 60)
    
    github_client = GithubClient(github_token=github_token, verbose=False)
    
    # Custom file processing callback
    def process_file_callback(file_path: str, file_size: int) -> tuple[bool, str]:
        """Custom logic to determine if a file should be processed."""
        # Skip large files
        if file_size > 1024 * 1024:  # 1MB
            return False, f"File too large: {file_size} bytes"
        
        # Skip certain file types
        skip_extensions = [".png", ".jpg", ".gif", ".ico", ".exe"]
        if any(file_path.endswith(ext) for ext in skip_extensions):
            return False, "Skipping binary/image files"
        
        return True, ""
    
    reader = GithubRepositoryReader(
        github_client=github_client,
        owner="microsoft",
        repo="vscode",
        use_parser=False,
        verbose=True,
        # Filter to only include specific directories
        filter_directories=(
            ["src", "extensions"], 
            GithubRepositoryReader.FilterType.INCLUDE
        ),
        # Exclude certain file extensions
        filter_file_extensions=(
            [".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".json"],
            GithubRepositoryReader.FilterType.EXCLUDE,
        ),
        # Custom processing callback
        process_file_callback=process_file_callback,
        fail_on_error=False,  # Continue processing if errors occur
        concurrent_requests=3,  # Limit concurrent requests
    )
    
    try:
        print("📖 Loading documents with advanced filtering...")
        documents = reader.load_data(branch="main")
        
        print(f"\n📊 Results with Filtering:")
        print(f"   Total documents created: {len(documents)}")
        
        # Analyze file types
        file_types = {}
        for doc in documents:
            if doc.extra_info and 'file_path' in doc.extra_info:
                ext = Path(doc.extra_info['file_path']).suffix or 'no_extension'
                file_types[ext] = file_types.get(ext, 0) + 1
        
        print(f"   File types processed:")
        for ext, count in sorted(file_types.items()):
            print(f"     {ext}: {count} files")
            
    except Exception as e:
        print(f"❌ Error reading repository with filtering: {e}")


def demo_issues_reader(github_token: str):
    """Demonstrate GitHub Issues reader."""
    print("\n" + "=" * 60)
    print("🐛 DEMO 3: GitHub Issues Reader")
    print("=" * 60)
    
    github_client = GitHubIssuesClient(github_token=github_token, verbose=True)
    
    reader = GitHubRepositoryIssuesReader(
        github_client=github_client,
        owner="toluwajibodu",  # Your actual repository
        repo="agentic-data-gen",
        verbose=True,
    )
    
    try:
        print("🐛 Loading issues...")
        
        # Load only open issues with specific labels
        documents = reader.load_data(
            state=GitHubRepositoryIssuesReader.IssueState.OPEN,
            labelFilters=[
                ("bug", GitHubRepositoryIssuesReader.FilterType.INCLUDE)
            ]
        )
        
        print(f"\n📊 Issues Results:")
        print(f"   Total bug issues found: {len(documents)}")
        
        # Show first few issues
        for i, doc in enumerate(documents[:3]):
            print(f"\n🐛 Issue {i+1}:")
            print(f"   ID: #{doc.doc_id}")
            print(f"   Content preview: {doc.text[:300]}...")
            if doc.extra_info:
                print(f"   State: {doc.extra_info.get('state', 'N/A')}")
                print(f"   Created: {doc.extra_info.get('created_at', 'N/A')}")
                print(f"   URL: {doc.extra_info.get('source', 'N/A')}")
                if 'labels' in doc.extra_info:
                    print(f"   Labels: {', '.join(doc.extra_info['labels'])}")
                    
    except Exception as e:
        print(f"❌ Error reading issues: {e}")


def demo_commits_reader(github_token: str):
    """Demonstrate GitHub Commits reader."""
    print("\n" + "=" * 60)
    print("🐛 DEMO 4: GitHub Commits Reader")
    print("=" * 60)

    github_client = GitHubCommitsClient(github_token=github_token, verbose=True)
    
    reader = GitHubRepositoryCommitsReader(
        github_client=github_client,
        owner="toluwajibodu",  # Your actual repository
        repo="agentic-data-gen",
        verbose=True,
    )

    try:
        print("🐛 Loading commits...")
        documents = reader.load_data()
        
        print(f"\n📊 Commits Results:")
        print(f"   Total commits found: {len(documents)}")

        # Show first few commits
        for i, doc in enumerate(documents[:3]):
            print(f"\n🐛 Commit {i+1}:")
            print(f"   ID: {doc.doc_id}")
            print(f"   Content preview: {doc.text[:300]}...")

            if doc.extra_info:
                print(f"   Author: {doc.extra_info.get('author_name', 'N/A')}")
                print(f"   Date: {doc.extra_info.get('author_date', 'N/A')}")
                print(f"   URL: {doc.extra_info.get('source', 'N/A')}")

    except Exception as e: 
        print(f"❌ Error reading commits: {e}")
        
        
        
        
        

def demo_pull_requests_reader(github_token: str):
    """Demonstrate GitHub Pull Requests reader."""
    print("\n" + "=" * 60)
    print("🔀 DEMO 5: GitHub Pull Requests Reader")
    print("=" * 60)
    
    github_client = GitHubPullRequestsClient(github_token=github_token, verbose=True)
    
    reader = GitHubRepositoryPullRequestsReader(
        github_client=github_client,
        owner="toluwajibodu",  # Your actual repository
        repo="agentic-data-gen",
        verbose=True,
    )
    
    try:
        print("🔀 Loading pull requests...")
        
        # Load all PRs with reviews
        documents = reader.load_data(
            state=GitHubRepositoryPullRequestsReader.PRState.ALL,
            max_prs=10,
            include_reviews=True,
        )
        
        print(f"\n📊 Pull Requests Results:")
        print(f"   Total PRs found: {len(documents)}")
        
        # Show first few PRs
        for i, doc in enumerate(documents[:3]):
            print(f"\n🔀 PR {i+1}:")
            print(f"   ID: #{doc.extra_info['number']}")
            print(f"   Title: {doc.extra_info['title']}")
            print(f"   Content preview: {doc.text[:300]}...")
            if doc.extra_info:
                print(f"   Author: {doc.extra_info.get('author', 'N/A')}")
                print(f"   State: {doc.extra_info.get('state', 'N/A')}")
                print(f"   Merged: {doc.extra_info.get('merged', False)}")
                
    except Exception as e:
        print(f"❌ Error reading pull requests: {e}")


def demo_collaborators_reader(github_token: str):
    """Demonstrate GitHub Collaborators reader."""
    print("\n" + "=" * 60)
    print("👥 DEMO 6: GitHub Collaborators Reader")
    print("=" * 60)
    
    github_client = GitHubCollaboratorsClient(github_token=github_token, verbose=True)
    
    reader = GitHubRepositoryCollaboratorsReader(
        github_client=github_client,
        owner="toluwajibodu",  # Your actual repository
        repo="agentic-data-gen",
        verbose=True,
    )
    
    try:
        print("👥 Loading collaborators...")
        documents = reader.load_data()
        
        print(f"\n📊 Collaborators Results:")
        print(f"   Total collaborators found: {len(documents)}")
        
        # Show collaborators
        for i, doc in enumerate(documents):
            print(f"\n👤 Collaborator {i+1}:")
            print(f"   Login: {doc.text}")
            if doc.extra_info:
                print(f"   Type: {doc.extra_info.get('type', 'N/A')}")
                print(f"   Role: {doc.extra_info.get('role_name', 'N/A')}")
                print(f"   Admin: {doc.extra_info.get('site_admin', 'N/A')}")
                
    except Exception as e:
        print(f"❌ Error reading collaborators: {e}")


def demo_document_analysis(documents: List[Document]):
    """Demonstrate basic document analysis."""
    print("\n" + "=" * 60)
    print("📊 DEMO 5: Document Analysis")
    print("=" * 60)
    
    if not documents:
        print("No documents to analyze.")
        return
    
    print(f"Total documents: {len(documents)}")
    
    # Analyze document sizes
    sizes = [len(doc.text) for doc in documents]
    print(f"Average document size: {sum(sizes) / len(sizes):.2f} characters")
    print(f"Largest document: {max(sizes)} characters")
    print(f"Smallest document: {min(sizes)} characters")
    
    # Sample content analysis
    print(f"\nSample document content:")
    if documents:
        doc = documents[0]
        print(f"Document ID: {doc.doc_id}")
        print(f"Content preview: {doc.text[:500]}...")


def main():
    """Main demonstration function."""
    print("🚀 GitHub Reader Comprehensive Demo")
    print("=" * 60)
    
    # Setup
    github_token = get_github_token()
    setup_event_handling()
    
    # Run demonstrations
    try:
        # Basic repository reading
        # demo_repository_reader_basic(github_token)
        
        # # Advanced repository reading (commented out due to potential size)
        # # demo_repository_reader_advanced(github_token)
        
        # # Issues reading
        # demo_issues_reader(github_token)
        
        # # Collaborators reading
        # demo_collaborators_reader(github_token)
        
        # Commits reading
        # demo_commits_reader(github_token)
        
        # Pull requests reading
        demo_pull_requests_reader(github_token)
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
