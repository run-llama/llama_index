# ChangeLog

## [Unreleased]

### New Features
- Added better support for vector store with existing data (e.g. allow configurable text key) for Pinecone and Weaviate. (#6393)
- Support batched upsert for Pineone (#6393)

### Bug Fixes
- None

### Breaking/Deprecated API Changes
- None

### Miscellaneous
- None

## [v0.6.23] - 2023-06-11

### Bug Fixes / Nits
- Remove hardcoded chunk size for citation query engine (#6408)
- Mongo demo improvements (#6406)
- Fix notebook (#6418)
- Cleanup RetryQuery notebook (#6381)

## [v0.6.22] - 2023-06-10

### New Features
- Added `SQLJoinQueryEngine` (generalization of `SQLAutoVectorQueryEngine`) (#6265)
- Added support for graph stores under the hood, and initial support for Nebula KG. More docs coming soon! (#2581)
- Added guideline evaluator to allow llm to provide feedback based on user guidelines (#4664)
- Added support for MongoDB Vector stores to enable Atlas knnbeta search (#6379)
- Added new CitationQueryEngine for inline citations of sources in response text (#6239)

### Bug Fixes
- Fixed bug with `delete_ref_doc` not removing all metadata from the docstore (#6192)
- FIxed bug with loading existing QDrantVectorStore (#6230)

### Miscellaneous 
- Added changelog officially to github repo (#6191)

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
