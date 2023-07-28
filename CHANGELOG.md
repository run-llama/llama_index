# ChangeLog

## [0.7.14] - 2023-07-28

### New Features
- Added HotpotQADistractor benchmark evaluator (#7034)
- Add metadata filter and delete support for LanceDB (#7048)
- Use MetadataFilters in opensearch (#7005)
- Added support for `KuzuGraphStore` (#6970)
- Added `kg_triplet_extract_fn` to customize how KGs are built (#7068)

### Bug Fixes / Nits
- Fix string formatting in context chat engine (#7050)
- Fixed tracing for async events (#7052)
- Less strict triplet extraction for KGs (#7059)
- Add configurable limit to KG data retrieved (#7059)
- Nebula connection improvements (#7059)
- Bug fix in building source nodes for agent response (#7067)

## [0.7.13] - 2023-07-26

### New Features
- Support function calling api for AzureOpenAI (#7041)

### Bug Fixes / Nits
- tune prompt to get rid of KeyError in SubQ engine (#7039)
- Fix validation of Azure OpenAI keys (#7042)

## [0.7.12] - 2023-07-25

### New Features
- Added `kwargs` to `ComposableGraph` for the underlying query engines (#6990)
- Validate openai key on init (#6940)
- Added async embeddings and async RetrieverQueryEngine (#6587)
- Added async `aquery` and `async_add` to PGVectorStore (#7031)
- Added `.source_nodes` attribute to chat engine and agent responses (#7029)
- Added `OpenInferenceCallback` for storing generation data in OpenInference format (#6998)

### Bug Fixes / Nits
- Fix achat memory initialization for data agents (#7000)
- Add `print_response_stream()` to agengt/chat engine response class (#7018)

## [v0.7.11.post1] - 2023-07-20

### New Features
- Default to pydantic question generation when possible for sub-question query engine (#6979)

### Bug Fixes / Nits
- Fix returned order of messages in large chat memory (#6979) 

## [v0.7.11] - 2023-07-19

### New Features
- Added a `SentenceTransformerRerank` node post-processor for fast local re-ranking (#6934)
- Add numpy support for evaluating queries in pandas query engine (#6935)
- Add metadata filtering support for Postgres Vector Storage integration (#6968)
- Proper llama2 support for agents and query engines (#6969)

### Bug Fixes / Nits
- Added `model_name` to LLMMetadata (#6911)
- Fallback to retriever service context in query engines (#6911)
- Fixed `as_chat_engine()` ValueError with extra kwargs (#6971

## [v0.7.10.post1] - 2023-07-18

### New Features
- Add support for Replicate LLM (vicuna & llama 2!)

### Bug Fixes / Nits
- fix streaming for condense chat engine (#6958)

## [v0.7.10] - 2023-07-17

### New Features
- Add support for chroma v0.4.0 (#6937)
- Log embedding vectors to callback manager (#6962)

### Bug Fixes / Nits
- add more robust embedding timeouts (#6779)
- improved connection session management on postgres vector store (#6843)

## [v0.7.9] - 2023-07-15

### New Features
- specify `embed_model="local"` to use default local embbeddings in the service context (#6806)
- Add async `acall` endpoint to `BasePydanticProgram` (defaults to sync version). Implement for `OpenAIPydanticProgram`

### Bug Fixes / Nits
- fix null metadata for searching existing vector dbs (#6912)
- add module guide docs for `SimpleDirectoryReader` (#6916)
- make sure `CondenseQuestionChatEngine` streaming chat endpoints work even if not explicitly setting `streaming=True` in the underlying query engine.

## [v0.7.8] - 2023-07-13

### New Features
- Added embedding speed benchmark (#6876)
- Added BEIR retrieval benchmark (#6825)

### Bug Fixes / Nits
- remove toctrees from deprecated_terms (#6895)
- Relax typing dependencies (#6879)
- docs: modification to evaluation notebook (#6840)
- raise error if the model does not support functions (#6896)
- fix(bench embeddings): bug not taking into account string length (#6899)x

## [v0.7.7] - 2023-07-13

### New Features
- Improved milvus consistency support and output fields support (#6452)
- Added support for knowledge graph querying w/ cypyer+nebula (#6642)
- Added `Document.example()` to create documents for fast prototyping (#6739)
- Replace react chat engine to use native reactive agent (#6870)

### Bug Fixes / Nits
- chore: added a help message to makefile (#6861)

### Bug Fixes / Nits
- Fixed support for using SQLTableSchema context_str attribute (#6891)

## [v0.7.6] - 2023-07-12

### New Features
- Added sources to agent/chat engine responses (#6854)
- Added basic chat buffer memory to agents / chat engines (#6857)
- Adding load and search tool (#6871)
- Add simple agent benchmark (#6869)
- add agent docs  (#6866)
- add react agent (#6865)

### Breaking/Deprecated API Changes
- Replace react chat engine with native react agent (#6870)
- Set default chat mode to "best": use openai agent when possible, otherwise use react agent (#6870)

### Bug Fixes / Nits
- Fixed support for legacy vector store metadata (#6867)
- fix chroma notebook in docs (#6872)
- update LC embeddings docs (#6868)

## [v0.7.5] - 2023-07-11

### New Features
- Add `Anthropic` LLM implementation (#6855)

### Bug Fixes / Nits
- Fix indexing error in `SentenceEmbeddingOptimizer` (#6850)
- fix doc for custom embedding model (#6851)
- fix(silent error): Add validation to `SimpleDirectoryReader` (#6819)
- Fix link in docs (#6833)
- Fixes Azure gpt-35-turbo model not recognized  (#6828)
- Update Chatbot_SEC.ipynb (#6808)
- Rename leftover original name to LlamaIndex (#6792)
- patch nested traces of the same type (#6791)

## [v0.7.4] - 2023-07-08

### New Features
- `MetadataExtractor` - Documnent Metadata Augmentation via LLM-based feature extractors (#6764)

### Bug Fixes / Nits
- fixed passing in query bundle to node postprocessors (#6780)
- fixed error in callback manager with nested traces (#6791)

## [v0.7.3] - 2023-07-07

### New Features
- Sub question query engine returns source nodes of sub questions in the callback manager (#6745)
- trulens integration (#6741)
- Add sources to subquestion engine (#6745)

### Bug Fixes / Nits
- Added/Fixed streaming support to simple and condense chat engines (#6717)
- fixed `response_mode="no_text"` response synthesizer (#6755)
- fixed error setting `num_output` and `context_window` in service context (#6766)
- Fix missing as_query_engine() in tutorial (#6747)
- Fixed variable sql_query_engine in the notebook (#6778)
- fix required function fields (#6761)
- Remove usage of stop token in Prompt, SQL gen (#6782)

## [v0.7.2] - 2023-07-06

### New Features
- Support Azure OpenAI (#6718)
- Support prefix messages (e.g. system prompt) in chat engine and OpenAI agent (#6723)
- Added `CBEventType.SUB_QUESTIONS` event type for tracking sub question queries/responses (#6716)

### Bug Fixes / Nits
- Fix HF LLM output error (#6737)
- Add system message support for langchain message templates (#6743)
- Fixed applying node-postprocessors (#6749)
- Add missing `CustomLLM` import under `llama_index.llms` (#6752)
- fix(typo): `get_transformer_tokenizer_fn` (#6729)
- feat(formatting): `black[jupyter]` (#6732)
- fix(test): `test_optimizer_chinese` (#6730)

## [v0.7.1] - 2023-07-05

### New Features
- Streaming support for OpenAI agents (#6694)
- add recursive retriever + notebook example (#6682)


## [v0.7.0] - 2023-07-04

### New Features
- Index creation progress bars (#6583)

### Bug Fixes/ Nits
- Improved chat refine template (#6645)

### Breaking/Deprecated API Changes

- Change `BaseOpenAIAgent` to use `llama_index.llms.OpenAI`. Adjust `chat_history` to use `List[ChatMessage]]` as type.
- Remove (previously deprecated) `llama_index.langchain_helpers.chain_wrapper` module.
- Remove (previously deprecated) `llama_index.token_counter.token_counter` module. See [migration guide](/how_to/callbacks/token_counting_migration.html) for more details on new callback based token counting.
- Remove `ChatGPTLLMPredictor` and `HuggingFaceLLMPredictor`. See [migration guide](/how_to/customization/llms_migration_guide.html) for more details on replacements.
- Remove support for setting `cache` via `LLMPredictor` constructor.
- Update `BaseChatEngine` interface:
  - adjust `chat_history` to use `List[ChatMessage]]` as type
  - expose `chat_history` state as a property
  - support overriding `chat_history` in `chat` and `achat` endpoints
- Remove deprecated arguments for `PromptHelper`: `max_input_size`, `embedding_limit`, `max_chunk_overlap`
- Update all notebooks to use native openai integration (#6696)

## [v0.6.38] - 2023-07-02

### New Features

- add optional tqdm progress during index creation (#6583)
- Added async support for "compact" and "refine" response modes (#6590)
- [feature]add transformer tokenize functionalities for optimizer (chinese) (#6659)
- Add simple benchmark for vector store (#6670)
- Introduce `llama_index.llms` module, with new `LLM` interface, and `OpenAI`, `HuggingFaceLLM`, `LangChainLLM` implementations. (#6615)
- Evaporate pydantic program (#6666)

### Bug Fixes / Nits

- Improve metadata/node storage and retrieval for RedisVectorStore (#6678)
- Fixed node vs. document filtering in vector stores (#6677)
- add context retrieval agent notebook link to docs (#6660)
- Allow null values for the 'image' property in the ImageNode class and se… (#6661)
- Fix broken links in docs (#6669)
- update milvus to store node content (#6667)

## [v0.6.37] - 2023-06-30

### New Features

- add context augmented openai agent (#6655)

## [v0.6.36] - 2023-06-29

### New Features

- Redis support for index stores and docstores (#6575)
- DuckDB + SQL query engine notebook (#6628)
- add notebook showcasing deplot data loader (#6638)

### Bug Fixes / Nits

- More robust JSON parsing from LLM for `SelectionOutputParser` (#6610)
- bring our loaders back in line with llama-hub (#6630)
- Remove usage of SQLStructStoreIndex in notebooks (#6585)
- MD reader: remove html tags and leave linebreaks alone (#6618)
- bump min langchain version to latest version (#6632)
- Fix metadata column name in postgres vector store (#6622)
- Postgres metadata fixes (#6626, #6634)
- fixed links to dataloaders in contribution.md (#6636)
- fix: typo in docs in creating custom_llm huggingface example (#6639)
- Updated SelectionOutputParser to handle JSON objects and arrays (#6610)
- Fixed docstring argument typo (#6652)

## [v0.6.35] - 2023-06-28

- refactor structured output + pydantic programs (#6604)

### Bug Fixes / Nits

- Fix serialization for OpenSearch vector stores (#6612)
- patch docs relationships (#6606)
- Bug fix for ignoring directories while parsing git repo (#4196)
- updated Chroma notebook (#6572)
- Backport old node name (#6614)
- Add the ability to change chroma implementation (#6601)

## [v0.6.34] - 2023-06-26

### Patch Update (v0.6.34.post1)

- Patch imports for Document obj for backwards compatibility (#6597)

### New Features

- New `TextNode`/`Document` object classes based on pydantic (#6586)
- `TextNode`/`Document` objects support metadata customization (metadata templates, exclude metadata from LLM or embeddings) (#6586)
- Nodes no longer require flat metadata dictionaries, unless the vector store you use requires it (#6586)

### Bug Fixes / Nits

- use `NLTK_DATA` env var to control NLTK download location (#6579)
- [discord] save author as metadata in group_conversations.py (#6592)
- bs4 -> beautifulsoup4 in requirements (#6582)
- negate euclidean distance (#6564)
- add df output parser notebook link to docs (#6581)

### Breaking/Deprecated API Changes

- `Node` has been renamed to `TextNode` and is imported from `llama_index.schema` (#6586)
- `TextNode` and `Document` must be instansiated with kwargs: `Document(text=text)` (#6586)
- `TextNode` (fka `Node`) has a `id_` or `node_id` property, rather than `doc_id` (#6586)
- `TextNode` and `Document` have a metadata property, which replaces the extra_info property (#6586)
- `TextNode` no longer has a `node_info` property (start/end indexes are accessed directly with `start/end_char_idx` attributes) (#6586)

## [v0.6.33] - 2023-06-25

### New Features

- Add typesense vector store (#6561)
- add df output parser (#6576)

### Bug Fixes / Nits

- Track langchain dependency via bridge module. (#6573)

## [v0.6.32] - 2023-06-23

### New Features

- add object index (#6548)
- add SQL Schema Node Mapping + SQLTableRetrieverQueryEngine + obj index fixes (#6569)
- sql refactor (NLSQLTableQueryEngine) (#6529)

### Bug Fixes / Nits

- Update vector_stores.md (#6562)
- Minor `BaseResponseBuilder` interface cleanup (#6557)
- Refactor TreeSummarize (#6550)

## [v0.6.31] - 2023-06-22

### Bug Fixes / Nits

- properly convert weaviate distance to score (#6545)
- refactor tree summarize and fix bug to not truncate context (#6550)
- fix custom KG retrieval notebook nits (#6551)

## [v0.6.30] - 2023-06-21

### New Features

- multi-selector support in router query engine (#6518)
- pydantic selector support in router query engine using OpenAI function calling API (#6518)
- streaming response support in `CondenseQuestionChatEngine` and `SimpleChatEngine` (#6524)
- metadata filtering support in `QdrantVectorStore` (#6476)
- add `PGVectorStore` to support postgres with pgvector (#6190)

### Bug Fixes / Nits

- better error handling in the mbox reader (#6248)
- Fix blank similarity score when using weaviate (#6512)
- fix for sorted nodes in `PrevNextNodePostprocessor` (#6048)

### Breaking/Deprecated API Changes

- Refactor PandasQueryEngine to take in df directly, deprecate PandasIndex (#6527)

## [v0.6.29] - 2023-06-20

### New Features

- query planning tool with OpenAI Function API (#6520)
- docs: example of kg+vector index (#6497)
- Set context window sizes for Cohere and AI21(J2 model) (#6485)

### Bug Fixes / Nits

- add default input size for Cohere and AI21 (#6485)
- docs: replace comma with colon in dict object (#6439)
- extra space in prompt and error message update (#6443)
- [Issue 6417] Fix prompt_templates docs page (#6499)
- Rip out monkey patch and update model to context window mapping (#6490)

## [v0.6.28] - 2023-06-19

### New Features

- New OpenAI Agent + Query Engine Cookbook (#6496)
- allow recursive data extraction (pydantic program) (#6503)

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

- Deprecated current token tracking (llm predictor and embed model will no longer track tokens in the future, please use the `TokenCountingCallback` (#6440)
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
