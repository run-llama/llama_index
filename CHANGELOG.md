# ChangeLog

## Unreleased

### Bug Fixes / Nits
- add default input size for Cohere and AI21 (#6485)

## [v0.6.28] - 2023-06-19

### New Features
- New OpenAI Agent + Query Engine Cookbook (#6496)
- allow recursive data extraction (pydantic program)  (#6503)

### Bug Fixes / Nits
- update mongo interface (#6501)
- fixes that we forgot to include for openai pydantic program (#6503) (#6504)
- Fix github pics in Airbyte notebook (#6493)

## [v0.6.27] - 2023-06-16

### New Features
- Add node doc_id filtering to weaviate (#6467)
- New `TokenCountingCallback` to customize and track embedding, prompt, and completion token usage (#6440)
- OpenAI Retrieval Function Agent (#6491)

### Breaking/Deprecated API Changes
- Deprecated current token tracking (llm predicotr and embed model will no longer track tokens in the future, please use the `TokenCountingCallback` (#6440)
- Add maximal marginal relevance to the Simple Vector Store, which can be enabled as a query mode (#6446)

### Bug Fixes / Nits
- `as_chat_engine` properly inherits the current service context (#6470)
- Use namespace when deleting from pinecone (#6475)
- Fix paths when using fsspec on windows (#3778)
- Fix for using custom file readers in `SimpleDirectoryReader` (#6477)
- Edit MMR Notebook (#6486)
- FLARE fixes (#6484)

## [v0.6.26] - 2023-06-14

### New Features
- Add OpenAIAgent and tutorial notebook for "build your own agent" (#6461)
- Add OpenAIPydanticProgram (#6462)

### Bug Fixes / Nits
- Fix citation engine import (#6456)

## [v0.6.25] - 2023-06-13

### New Features
- Added FLARE query engine (#6419).

## [v0.6.24] - 2023-06-12

### New Features
- Added better support for vector store with existing data (e.g. allow configurable text key) for Pinecone and Weaviate. (#6393)
- Support batched upsert for Pineone (#6393)
- Added initial [guidance](https://github.com/microsoft/guidance/) integration. Added `GuidancePydanticProgram` for generic structured output generation and `GuidanceQuestionGenerator` for generating sub-questions in `SubQuestionQueryEngine` (#6246).

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
