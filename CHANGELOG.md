# Changelog

## [Unreleased]

### New Features
- Added support for graph stores under the hood, and initial support for Nebula KG. More docs coming soon!
- Added guideline evaluator to allow llm to provide feedback based on user guidelines

### Bug Fixes
- None

### Breaking/Deprecated API Changes
- None

### Miscellaneous 
- Added changelog officially to github repo

## [v0.6.21] - 2023-06-06

### New Features
- SimpleDirectoryReader has new `filename_as_id` flag to automatically set the doc_id (useful for `refresh_ref_docs()`)
- DocArray vector store integration
- Tair vector store integration
- Weights and Biases callback handler for tracing and versioning indexes
- Can initialize indexes directly from a vector store: `index = VectorStoreIndex.from_vector_store(vector_store=vector_store)`

### Bug Fixes
- Fixed multimodal notebook
- Updated/fixed the SQL tutorial in the docs
 
### Miscellaneous 
- Minor docs updates
- Added github pull-requset templates
- Added github issue-forms

## [v0.6.20] - 2023-06-04

### New Features
- Added new JSONQueryEngine that uses JSON schema to deliver more accurate JSON query answers
- Metadata support for redis vector-store
- Added Supabase vector store integration

### Bug Fixes
- Fixed typo in text-to-sql prompt

### Breaking/Deprecated API Changes
- Removed GPT prefix from indexes (old imports/names are still supported though)
 
### Miscellaneous 
- Major docs updates, brought important modules to the top level

## [v0.6.19] - 2023-06-02

### New Features
- Added agent tool abstraction for llama-hub data loaders
 
### Miscellaneous 
- Minor doc updates

## [v0.6.18] - 2023-06-02

### Miscellaneous 
- Added `Discover LlamaIndex` video series to the tutorials docs section
- Minor docs updates
