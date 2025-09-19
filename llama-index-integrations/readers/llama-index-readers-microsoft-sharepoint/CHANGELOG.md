# CHANGELOG

## [0.7.0] - 2025-01-15

### 🎉 Major Release - Enhanced SharePoint Integration

#### ✨ New Features
- **📄 SharePoint Page Reading**: Complete support for loading SharePoint site pages as documents
  - Use `sharepoint_type=SharePointType.PAGE` to load pages instead of files
  - Support for both all pages and specific page loading via `page_name`
  - Full HTML content extraction with metadata

- **🔧 Custom File Parsers**: Advanced file parsing system
  - Support for specialized parsers: PDF, DOCX, PPTX, HTML, CSV, Excel, Images, JSON, TXT
  - `CustomParserManager` for efficient parser management
  - Automatic file type detection and parser selection
  - Complete file parser implementations in `file_parsers.py`

- **📊 Event System**: Real-time processing monitoring
  - Comprehensive event classes: `PageDataFetchStartedEvent`, `PageDataFetchCompletedEvent`, `PageSkippedEvent`, `PageFailedEvent`, `TotalPagesToProcessEvent`
  - Integration with LlamaIndex instrumentation system
  - Event dispatching for monitoring document processing progress

- **🎯 Document Callbacks**: Advanced filtering and processing
  - `process_document_callback` for custom document filtering logic
  - `process_attachment_callback` for attachment handling
  - Flexible callback system for custom processing workflows

- **⚙️ Enhanced Error Handling**: Configurable error behavior
  - `fail_on_error` parameter for controlling error handling strategy
  - Option to continue processing when individual files fail
  - Improved error reporting and logging

#### 🛠️ Technical Improvements
- **Type Safety**: Complete FileType enum with all supported formats
- **Code Organization**: Modular architecture with separate event and parser modules  
- **Test Coverage**: Comprehensive test suite with 27+ test scenarios
- **Documentation**: Extensive README with examples and configuration options
- **Performance**: Optimized file processing and memory management

#### 🔧 Breaking Changes
- Constructor signature updated to support new parameters
- `sharepoint_type` parameter added (defaults to `SharePointType.DRIVE` for backward compatibility)
- `custom_parsers` requires `custom_folder` parameter when used
- Event system integration may require dispatcher setup for monitoring

#### 📦 Dependencies
- Added optional `[file_parsers]` extra for enhanced file processing capabilities
- Updated core dependencies for better compatibility
- Support for Python 3.9+

## [0.5.1] - 2025-04-02

- Fix issue with folder path encoding when a file path contains special characters

## [0.1.7] - 2024-04-03

- Use recursive strategy by default for reading from a folder

## [0.1.6] - 2024-04-01

- Allow passing arguments for sitename and folder path during construction of reader

## [0.1.4] - 2024-03-26

- Make the reader serializable by inheriting from `BasePydanticReader` instead of `BaseReader`.

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
