# CHANGELOG

## [0.4.0] - 2025-09-22

### Added

- **Fast Mode Parameter**: Added `fast_mode` parameter to enable faster but shorter search results

  - Can be set during `ValyuToolSpec` initialization for default behavior
  - Can be overridden in individual search calls
  - User-configurable but not exposed to LLM models for direct control
  - Good for general purpose queries where speed is prioritized over comprehensiveness

- **Contents API Integration**: Added comprehensive URL content extraction functionality
  - New `get_contents()` method in `ValyuToolSpec` for extracting content from URLs
  - New `ValyuRetriever` class for integrating URL content extraction with LlamaIndex retrieval pipelines
  - Model can only specify URLs - all extraction parameters controlled by user
  - Supports AI-powered summarization and structured data extraction
  - Configurable extraction effort, response length, and cost limits
  - Automatic URL parsing from natural language queries

### Updated

- Updated Valyu SDK to version 2.2.1 for fast mode and contents API support
- Enhanced test coverage with fast mode parameter testing and comprehensive contents API tests
- Updated examples to demonstrate fast mode usage and contents API functionality
- Added retriever examples and comprehensive API demonstrations

## [0.3.1] - Previous Release

### Features

- Enhanced search parameters support
- Updated SDK integration

## [0.1.0] - 2025-02-11

- Initial release
