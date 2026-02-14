# Plugin Marketplace

LlamaIndex now supports registering external repositories as plugin marketplaces. This allows you to install packs/skills from custom sources beyond the default LlamaIndex repository.

## Overview

The plugin marketplace feature enables:
- Registering external GitHub repositories as plugin sources
- Installing plugins from multiple marketplaces
- Managing marketplace configurations

## Built-in Marketplaces

The following marketplaces are registered by default:

| Name | Repository | Description |
|------|-----------|-------------|
| `llamaindex` | `run-llama/llama_index` | Official LlamaIndex packs repository |
| `superpowers-marketplace` | `obra/superpowers-marketplace` | Curated Claude Code plugin marketplace by obra |

## Quick Start

### Using the `plugin` CLI (Recommended)

The `plugin` command provides a unified interface for managing marketplaces and installing plugins.

#### 1. Add a Marketplace

```bash
llamaindex-cli plugin marketplace add obra/superpowers-marketplace \
  --name superpowers-marketplace \
  --description "Curated Claude Code plugin marketplace by obra"
```

#### 2. List Marketplaces

```bash
llamaindex-cli plugin marketplace list
```

#### 3. Install a Plugin

```bash
llamaindex-cli plugin install superpowers@superpowers-marketplace
```

The format is: `<PluginName>@<marketplace-name>`

#### 4. Remove a Marketplace

```bash
llamaindex-cli plugin marketplace remove superpowers-marketplace
```

**Note:** The default `llamaindex` marketplace cannot be removed.

### Example: Installing Superpowers

The [obra/superpowers-marketplace](https://github.com/obra/superpowers-marketplace) is a curated Claude Code plugin marketplace that includes skills for brainstorming, planning, and executing implementation tasks.

```bash
# The superpowers-marketplace is pre-registered, so just install directly:
llamaindex-cli plugin install superpowers@superpowers-marketplace

# Or install specific skill packs:
llamaindex-cli plugin install SuperpowersBrainstormPack@superpowers-marketplace --download-dir ./skills
llamaindex-cli plugin install SuperpowersWritePlanPack@superpowers-marketplace --download-dir ./skills
llamaindex-cli plugin install SuperpowersExecutePlanPack@superpowers-marketplace --download-dir ./skills
```

Available superpowers skills:
- **brainstorm** - Interactive design refinement before writing code
- **write-plan** - Create structured implementation plans
- **execute-plan** - Execute plans in batches with TDD methodology

### Legacy CLI Commands

The original `marketplace` and `download-llamapack` commands are still supported:

```bash
# Register a marketplace
llamaindex-cli marketplace add huggingface/skills \
  --name huggingface-skills \
  --base-path "" \
  --description "HuggingFace Skills Repository"

# List marketplaces
llamaindex-cli marketplace list

# Install a pack
llamaindex-cli download-llamapack <SkillName>@huggingface-skills --download-dir ./my_skills

# Remove a marketplace
llamaindex-cli marketplace remove huggingface-skills
```

**Arguments:**
- `repository` (required): GitHub repository in format `owner/repo`
- `--name` or `-n` (required): Short name for the marketplace (used in install commands)
- `--branch` or `-b` (optional): Git branch to use (default: `main`)
- `--base-path` or `-p` (optional): Base path within the repository for packs
- `--description` or `-d` (optional): Human-readable description

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
    "name": "superpowers-marketplace",
    "repository": "obra/superpowers-marketplace",
    "branch": "main",
    "base_path": "",
    "description": "Curated Claude Code plugin marketplace by obra"
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

# Download from superpowers marketplace
pack_cls = download_llama_pack(
    llama_pack_class="SuperpowersPack@superpowers-marketplace",
    download_dir="./my_plugins"
)

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
**Solution**: Register the marketplace using `llamaindex-cli plugin marketplace add` or `llamaindex-cli marketplace add`

### Cannot remove default marketplace
```
Error: Marketplace 'llamaindex' not found or cannot be removed
```
**Solution**: The default `llamaindex` marketplace cannot be removed

### Pack not found in marketplace
```
Failed to find python package for class PackName
```
**Solution**: Ensure the pack exists in the marketplace repository and follows the correct structure

## Advanced Usage

### Custom Branch
```bash
llamaindex-cli plugin marketplace add org/repo \
  --name my-marketplace \
  --branch develop
```

### Custom Base Path
If packs are in a subdirectory:
```bash
llamaindex-cli plugin marketplace add org/repo \
  --name my-marketplace \
  --base-path custom/packs/directory
```

### Multiple Marketplaces
You can register multiple marketplaces and install packs from any of them:
```bash
llamaindex-cli plugin install Pack1@marketplace-a
llamaindex-cli plugin install Pack2@marketplace-b
llamaindex-cli plugin install SuperpowersPack@superpowers-marketplace
llamaindex-cli download-llamapack Pack3 --download-dir ./dir3  # Uses default
```
