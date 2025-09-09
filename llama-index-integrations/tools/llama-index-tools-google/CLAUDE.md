# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LlamaIndex tools integration package (`llama-index-tools-google`) that provides Google services integration for LlamaIndex agents. It includes three main tool specifications:

1. **GoogleSearchToolSpec** - Custom Google search functionality using Google Custom Search API
2. **GmailToolSpec** - Gmail email reading, searching, drafting, and sending capabilities
3. **GoogleCalendarToolSpec** - Google Calendar event reading and creation

## Development Commands

### Testing

```bash
make test              # Run all tests via pytest
pytest tests          # Direct pytest invocation
```

### Linting and Formatting

```bash
make lint             # Run pre-commit hooks (black, ruff, codespell) and mypy
make format           # Run black autoformatter only
```

### Documentation

```bash
make watch-docs       # Build and watch documentation with sphinx-autobuild
```

## Code Architecture

### Tool Specifications Structure

Each Google service is implemented as a separate tool spec inheriting from `BaseToolSpec`:

- **Base classes**: All tools extend `llama_index.core.tools.tool_spec.base.BaseToolSpec`
- **Authentication**: Gmail and Calendar tools use OAuth 2.0 with local credential files (`credentials.json` and `token.json`)
- **Search tool**: Uses API key authentication with Google Custom Search API

### Key Components

#### GmailToolSpec (`llama_index/tools/google/gmail/base.py`)

- **Functions**: `load_data`, `search_messages`, `create_draft`, `update_draft`, `get_draft`, `send_draft`
- **Authentication**: OAuth 2.0 with Gmail API scopes for compose and readonly access
- **Message parsing**: Supports both HTML (BeautifulSoup) and iterative plain text extraction

#### GoogleCalendarToolSpec (`llama_index/tools/google/calendar/base.py`)

- **Functions**: `load_data`, `create_event`, `get_date`
- **Authentication**: OAuth 2.0 with Calendar API scope
- **Event handling**: Loads upcoming events and creates new calendar events with attendees

#### GoogleSearchToolSpec (`llama_index/tools/google/search/base.py`)

- **Functions**: `google_search`, `agoogle_search` (async version)
- **Authentication**: API key based
- **Configuration**: Requires custom search engine ID and supports result count limits (1-10)

### Authentication Requirements

- **Gmail/Calendar**: Requires `credentials.json` OAuth client file in project root
- **Search**: Requires Google Custom Search API key and custom search engine ID
- **Token storage**: OAuth tokens automatically saved to `token.json`

## Development Notes

- Uses `hatchling` as build backend
- Python 3.9+ required
- Pre-commit hooks configured for black, ruff, codespell, and mypy
- Tests located in `tests/` directory
- Examples provided in Jupyter notebooks under `examples/`

## Known Issues

- Calendar tool may fail to create events if timezone cannot be inferred by the agent
- OAuth authentication requires manual approval each time tools are invoked
