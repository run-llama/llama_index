# Plugin Marketplace

LlamaIndex now supports registering external repositories as plugin marketplaces. This allows you to install packs/skills from custom sources beyond the default LlamaIndex repository.

## Overview

The plugin marketplace feature enables:
- Registering external GitHub repositories as plugin sources
- Installing plugins from multiple marketplaces
- Managing marketplace configurations

## Quick Start

### 1. Register a Marketplace

Register an external repository as a plugin marketplace:

```bash
llamaindex-cli marketplace add huggingface/skills \
  --name huggingface-skills \
  --base-path "" \
  --description "HuggingFace Skills Repository"
```

**Arguments:**
- `repository` (required): GitHub repository in format `owner/repo`
- `--name` or `-n` (required): Short name for the marketplace (used in install commands)
- `--branch` or `-b` (optional): Git branch to use (default: `main`)
- `--base-path` or `-p` (optional): Base path within the repository for packs
- `--description` or `-d` (optional): Human-readable description

### 2. List Marketplaces

View all registered marketplaces:

```bash
llamaindex-cli marketplace list
```

Output example:
```
Registered marketplaces:

  llamaindex
    Repository: run-llama/llama_index
    Branch: main
    Base path: llama-index-packs
    Description: Official LlamaIndex packs repository

  huggingface-skills
    Repository: huggingface/skills
    Branch: main
    Description: HuggingFace Skills Repository
```

### 3. Install a Plugin from a Marketplace

Install a plugin using the marketplace specifier:

```bash
llamaindex-cli download-llamapack <SkillName>@huggingface-skills --download-dir ./my_skills
```

The format is: `<PackName>@<marketplace-name>`

**Examples:**
```bash
# Install from HuggingFace marketplace
llamaindex-cli download-llamapack TextGenerationSkill@huggingface-skills --download-dir ./skills

# Install from default LlamaIndex marketplace (no @ specifier needed)
llamaindex-cli download-llamapack GmailOpenAIAgentPack --download-dir ./packs
```

### 4. Remove a Marketplace

Remove a registered marketplace:

```bash
llamaindex-cli marketplace remove huggingface-skills
```

**Note:** The default `llamaindex` marketplace cannot be removed.

## Configuration

Marketplace configurations are stored in `~/.llamaindex/marketplaces.json`.

Example configuration:
```json
[
  {
    "name": "llamaindex",
    "repository": "run-llama/llama_index",
    "branch": "main",
    "base_path": "llama-index-packs",
    "description": "Official LlamaIndex packs repository"
  },
  {
    "name": "huggingface-skills",
    "repository": "huggingface/skills",
    "branch": "main",
    "base_path": "",
    "description": "HuggingFace Skills Repository"
  }
]
```

## How It Works

1. **Marketplace Registration**: When you register a marketplace, the configuration is saved to `~/.llamaindex/marketplaces.json`

2. **Pack Discovery**: The system constructs GitHub raw content URLs based on the marketplace configuration

3. **Pack Installation**: When you install a pack with `@marketplace-name`, the system:
   - Parses the pack identifier to extract the marketplace name
   - Looks up the marketplace configuration
   - Downloads the pack from the specified GitHub repository
   - Installs dependencies from the pack's `pyproject.toml`

## Repository Structure Requirements

For a repository to work as a marketplace, it should follow this structure:

```
repository/
├── llama-index-packs-{pack-name}/
│   ├── llama_index/
│   │   └── packs/
│   │       └── {snake_case_name}/
│   │           ├── __init__.py
│   │           ├── base.py
│   │           └── ...
│   ├── pyproject.toml
│   ├── README.md
│   └── tests/
```

Alternatively, if you use `--base-path`, the structure should be:

```
repository/
└── {base-path}/
    └── llama-index-packs-{pack-name}/
        └── ...
```

## Example: Setting Up HuggingFace Skills

```bash
# 1. Register the marketplace
llamaindex-cli marketplace add huggingface/skills \
  --name huggingface-skills \
  --description "HuggingFace Skills Repository"

# 2. List to verify
llamaindex-cli marketplace list

# 3. Install a skill
llamaindex-cli download-llamapack MySkill@huggingface-skills --download-dir ./skills
```

## Python API

You can also use the marketplace manager programmatically:

```python
from llama_index.core.llama_pack.marketplace import MarketplaceManager

# Create manager instance
manager = MarketplaceManager()

# Add a marketplace
manager.add_marketplace(
    name="custom-marketplace",
    repository="org/repo",
    branch="main",
    base_path="packs",
    description="My custom marketplace"
)

# List marketplaces
marketplaces = manager.list_marketplaces()
for m in marketplaces:
    print(f"{m.name}: {m.repository}")

# Get marketplace info
marketplace = manager.get_marketplace("custom-marketplace")
url = manager.get_marketplace_url("custom-marketplace")

# Remove marketplace
manager.remove_marketplace("custom-marketplace")
```

## Downloading Packs Programmatically

```python
from llama_index.core.llama_pack.download import download_llama_pack

# Download from custom marketplace
pack_cls = download_llama_pack(
    llama_pack_class="MyPackName@custom-marketplace",
    download_dir="./my_packs"
)

# Download from default marketplace (no @ specifier)
pack_cls = download_llama_pack(
    llama_pack_class="GmailOpenAIAgentPack",
    download_dir="./packs"
)
```

## Troubleshooting

### Marketplace not found
```
Error: Marketplace 'xyz' not found. Register it first with: llamaindex-cli marketplace add
```
**Solution**: Register the marketplace using `llamaindex-cli marketplace add`

### Cannot remove default marketplace
```
Error: Marketplace 'llamaindex' not found or cannot be removed
```
**Solution**: The default LlamaIndex marketplace cannot be removed

### Pack not found in marketplace
```
Failed to find python package for class PackName
```
**Solution**: Ensure the pack exists in the marketplace repository and follows the correct structure

## Advanced Usage

### Custom Branch
```bash
llamaindex-cli marketplace add org/repo \
  --name my-marketplace \
  --branch develop
```

### Custom Base Path
If packs are in a subdirectory:
```bash
llamaindex-cli marketplace add org/repo \
  --name my-marketplace \
  --base-path custom/packs/directory
```

### Multiple Marketplaces
You can register multiple marketplaces and install packs from any of them:
```bash
llamaindex-cli download-llamapack Pack1@marketplace-a --download-dir ./dir1
llamaindex-cli download-llamapack Pack2@marketplace-b --download-dir ./dir2
llamaindex-cli download-llamapack Pack3 --download-dir ./dir3  # Uses default
```
