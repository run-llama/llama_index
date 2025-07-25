# LlamaIndex Readers Integration: ORCID

## Overview

Reads researcher profiles from [ORCID](https://orcid.org). Unlike other readers that focus on papers, this gets researcher info like their bio, employment history, and publication list.

## Features

- Loads researcher profiles by ORCID ID
- Gets name, bio, keywords, employment/education history  
- Includes linked publications
- Validates ORCID IDs with checksum verification
- No API key needed

## Installation

```bash
pip install llama-index-readers-orcid
```

## Usage

### Basic Usage

```python
from llama_index.readers.orcid import ORCIDReader

# Initialize reader
reader = ORCIDReader()

# Load researcher profiles
documents = reader.load_data(orcid_ids=["0000-0002-1825-0097"])

# Process documents
for doc in documents:
    print(f"Researcher: {doc.metadata['orcid_id']}")
    print(doc.text)
```

### Advanced Configuration

```python
# Use sandbox environment for testing
reader = ORCIDReader(
    sandbox=True,              # Use test environment
    include_works=True,        # Include publications
    include_employment=True,   # Include job history
    include_education=True,    # Include education
    max_works=20,             # Limit publications per researcher
    rate_limit_delay=1.0,     # Seconds between API calls
    timeout=30                # Request timeout in seconds
)
```

## Use Cases

### 1. Building Researcher Expertise Databases
```python
# Find researchers in specific fields
expertise_index = VectorStoreIndex.from_documents(documents)
experts = expertise_index.query("machine learning researchers at Harvard")
```

### 2. Academic Network Analysis
```python
# Analyze collaboration networks through shared affiliations
# and co-authored works
```

### 3. Grant and Funding Intelligence
```python
# Track researcher funding history and grant success
```

### 4. Academic Recruiting
```python
# Match researcher profiles with job requirements
```

## Why ORCID Reader?

Other readers (ArXiv, PubMed) focus on papers. This reader gets researcher info - who they are, where they work, their career history.

## API Reference

### Constructor Parameters

- `sandbox` (bool): Use ORCID sandbox environment. Default: False
- `include_works` (bool): Include research publications. Default: True
- `include_employment` (bool): Include employment history. Default: True
- `include_education` (bool): Include education history. Default: True
- `max_works` (int): Maximum publications per researcher. Default: 50
- `rate_limit_delay` (float): Delay between API requests in seconds. Default: 0.5
- `timeout` (int): Request timeout in seconds. Default: 30

### Methods

`load_data(orcid_ids)` - Takes a list of ORCID IDs and returns Document objects with the researcher profiles.

## Data Format

Each document contains structured text with:
- ORCID ID
- Name and biography
- Research keywords
- External identifiers (Scopus, ResearcherID, etc.)
- Website URLs
- Employment history with dates
- Education history with dates
- Research works (if included)

## Error Handling

Handles invalid ORCID IDs, private profiles, and API rate limits automatically.

## Examples

See [examples/orcid_demo_with_outputs.ipynb](examples/orcid_demo_with_outputs.ipynb) for usage examples.

## Limitations

Only gets public data. Some profiles might be private or incomplete.

## Contributing

This reader is part of the LlamaIndex community. Contributions and improvements are welcome!