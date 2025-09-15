# GoogleDriveReaderV2 - Optimized Performance

## Overview

`GoogleDriveReaderV2` is an optimized version of the original `GoogleDriveReader` that significantly reduces Google Drive API calls and improves performance when working with large numbers of files.

## Key Optimizations

### 1. Batch File Information Retrieval

- **Before**: Individual `get()` API calls for each file to retrieve metadata
- **After**: Comprehensive `list()` API calls with expanded fields to get all file information in fewer requests
- **Impact**: Reduces API calls from O(n) to O(log n) for file metadata

### 2. Metadata Caching

- **Before**: Redundant API calls to fetch the same file information multiple times
- **After**: Internal cache stores file metadata to eliminate redundant API calls
- **Impact**: Near-zero additional API calls for repeated operations

### 3. Optimized Path Resolution

- **Before**: Individual parent folder lookups for each file to build full paths
- **After**: Batch folder hierarchy caching and path building from list responses
- **Impact**: Dramatically reduces API calls for path resolution

### 4. Efficient File Downloads

- **Before**: Additional API call before each download to get file details
- **After**: Uses cached metadata, eliminating redundant API calls
- **Impact**: One less API call per file download

## Performance Improvements

For a folder with 100 files:

- **Original**: ~200+ API calls (2+ per file for metadata and path resolution)
- **V2**: ~10-20 API calls (batch operations with pagination)
- **Improvement**: 90%+ reduction in API calls

## Usage

The V2 reader is a drop-in replacement for the original reader:

```python
# Import the V2 reader
from llama_index.readers.google import GoogleDriveReaderV2

# Use exactly like the original reader
reader = GoogleDriveReaderV2(
    folder_id="your-folder-id",
    service_account_key_path="path/to/service_account.json",
)

# All original methods work the same way
documents = reader.load_data()
resources = reader.list_resources()
info = reader.get_resource_info("file-id")
```

## New Features

### Cache Management

```python
# Clear internal caches (useful for long-running applications)
reader.clear_cache()
```

### Performance Monitoring

The V2 reader maintains internal caches that you can inspect:

- `reader._file_metadata_cache`: Cached file metadata
- `reader._folder_path_cache`: Cached folder path information

## Backward Compatibility

`GoogleDriveReaderV2` is fully backward compatible with `GoogleDriveReader`:

- Same constructor parameters
- Same public methods and return types
- Same error handling behavior
- Extends the base class, so all existing functionality is preserved

## When to Use V2

Use `GoogleDriveReaderV2` when:

- Working with folders containing many files (>10)
- Making multiple operations on the same files
- Performance and API quota usage are concerns
- You want the latest optimizations without changing your code

The original `GoogleDriveReader` remains available for:

- Simple single-file operations
- Backward compatibility requirements
- When you prefer the original implementation

## Implementation Details

### Optimized Methods

- `_get_fileids_meta_optimized()`: Batch file listing with comprehensive metadata
- `_download_file_optimized()`: Download using cached metadata
- `_build_folder_path_cache()`: Efficient folder hierarchy caching
- `_get_relative_path_optimized()`: Path resolution using cached data

### Caching Strategy

- **File Metadata Cache**: Stores complete file information from API responses
- **Folder Path Cache**: Stores computed folder paths to avoid repeated traversals
- **Automatic Cache Population**: Caches are populated during normal operations
- **Manual Cache Management**: `clear_cache()` method for explicit cache control

### Error Handling

- Same error handling as the original reader
- Graceful fallback to original methods when cache misses occur
- Preserves all existing error reporting and logging behavior

## Migration Guide

To migrate from `GoogleDriveReader` to `GoogleDriveReaderV2`:

1. Change the import:

   ```python
   # Before
   from llama_index.readers.google import GoogleDriveReader

   # After
   from llama_index.readers.google import GoogleDriveReaderV2
   ```

2. Update the instantiation:

   ```python
   # Before
   reader = GoogleDriveReader(...)

   # After
   reader = GoogleDriveReaderV2(...)
   ```

3. No other changes needed - all methods work identically!

## API Quota Considerations

The V2 reader is designed to be much more efficient with Google Drive API quotas:

- Significantly fewer API calls for the same operations
- Better suited for applications with quota constraints
- Reduced likelihood of hitting rate limits
- More efficient for batch processing scenarios
