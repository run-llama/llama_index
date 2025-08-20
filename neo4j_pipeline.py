#!/usr/bin/env python3
"""GitHub → PropertyGraphIndex → Neo4j Pipeline"""

import os
import sys
from pathlib import Path

# Add the github reader to path
sys.path.insert(0, str(Path(__file__).parent / "llama-index-integrations/readers/llama-index-readers-github"))

from llama_index.readers.github import (
    GithubRepositoryReader, GithubClient,
    GitHubRepositoryIssuesReader, GitHubIssuesClient,
    GitHubRepositoryCommitsReader, GitHubCommitsClient,
    GitHubRepositoryPullRequestsReader, GitHubPullRequestsClient,
    # GitHubRepositoryCollaboratorsReader, GitHubCollaboratorsClient,  # Disabled due to nested data types
)
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Configuration - All sensitive values from environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
NEO4J_URI = os.environ.get("NEO4J_URI") 
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# Validate required environment variables
missing_vars = []
if not GITHUB_TOKEN:
    missing_vars.append("GITHUB_TOKEN")
if not NEO4J_URI:
    missing_vars.append("NEO4J_URI")
if not NEO4J_PASSWORD:
    missing_vars.append("NEO4J_PASSWORD")

if missing_vars:
    print("❌ Missing required environment variables:")
    for var in missing_vars:
        print(f"   {var}")
    print("\nPlease set them before running:")
    print("export GITHUB_TOKEN='your_github_token'")
    print("export NEO4J_URI='neo4j+ssc://your_neo4j_url'")
    print("export NEO4J_PASSWORD='your_neo4j_password'")
    sys.exit(1)

NEO4J_AUTH = ("neo4j", NEO4J_PASSWORD)
REPO_OWNER = "toluwajibodu"
REPO_NAME = "agentic-data-gen"

print("🚀 GitHub → PropertyGraphIndex → Neo4j Pipeline")

# 1. Setup Neo4j Property Graph Store
print("🔧 Setting up Neo4j store...")
neo4j_store = Neo4jPropertyGraphStore(
    username=NEO4J_AUTH[0],
    password=NEO4J_AUTH[1], 
    url=NEO4J_URI
)

# 2. Collect documents from all GitHub sources
all_documents = []

# Repository files
print("📁 Loading repository files...")
repo_reader = GithubRepositoryReader(
    github_client=GithubClient(github_token=GITHUB_TOKEN),
    owner=REPO_OWNER, repo=REPO_NAME
)
all_documents.extend(repo_reader.load_data(branch="main"))

# Issues
print("🐛 Loading issues...")
issues_reader = GitHubRepositoryIssuesReader(
    github_client=GitHubIssuesClient(github_token=GITHUB_TOKEN),
    owner=REPO_OWNER, repo=REPO_NAME
)
all_documents.extend(issues_reader.load_data())

# Commits
print("📝 Loading commits...")
commits_reader = GitHubRepositoryCommitsReader(
    github_client=GitHubCommitsClient(github_token=GITHUB_TOKEN),
    owner=REPO_OWNER, repo=REPO_NAME
)
all_documents.extend(commits_reader.load_data())

# Pull Requests
print("🔀 Loading pull requests...")
prs_reader = GitHubRepositoryPullRequestsReader(
    github_client=GitHubPullRequestsClient(github_token=GITHUB_TOKEN),
    owner=REPO_OWNER, repo=REPO_NAME
)
all_documents.extend(prs_reader.load_data(max_prs=10))

# Collaborators (temporarily disabled due to nested permission objects)
# print("👥 Loading collaborators...")
# collab_reader = GitHubRepositoryCollaboratorsReader(
#     github_client=GitHubCollaboratorsClient(github_token=GITHUB_TOKEN),
#     owner=REPO_OWNER, repo=REPO_NAME
# )
# all_documents.extend(collab_reader.load_data())

# 3. Create Property Graph Index with Neo4j
print(f"🏗️ Creating Property Graph with {len(all_documents)} documents...")
index = PropertyGraphIndex.from_documents(
    all_documents,
    property_graph_store=neo4j_store,
    embed_model="local",  # Use local embeddings instead of OpenAI
    show_progress=True
)

print("✅ Property Graph created and stored in Neo4j!")
print(f"📊 Total documents processed: {len(all_documents)}")
print("🔍 You can now query the graph using index.as_query_engine()")
