# ChangeLog

## [0.9.22] - 2023-12-26

### New Features

- Added `.iter_data()` method to `SimpleDirectoryReader` (#9658)
- Added async support to `Ollama` LLM (#9689)
- Expanding pinecone filter support for `in` and `not in` (#9683)

### Bug Fixes / Nits

- Improve BM25Retriever performance (#9675)
- Improved qdrant hybrid search error handling (#9707)
- Fixed `None` handling in `ChromaVectorStore` (#9697)
- Fixed postgres schema creation if not existing (#9712)

## [0.9.21] - 2023-12-23

### New Features

- Added zilliz cloud as a managed index (#9605)

### Bug Fixes / Nits

- Bedrock client and LLM fixes (#9671, #9646)

## [0.9.20] - 2023-12-21

### New Features

- Added `insert_batch_size` to limit number of embeddings held in memory when creating an index, defaults to 2048 (#9630)
- Improve auto-retrieval (#9647)
- Configurable Node ID Generating Function (#9574)
- Introduced action input parser (#9575)
- qdrant sparse vector support (#9644)
- Introduced upserts and delete in ingestion pipeline (#9643)
- Add Zilliz Cloud Pipeline as a Managed Index (#9605)
- Add support for Google Gemini models via VertexAI (#9624)
- support allowing additional metadata filters on autoretriever (#9662)

### Bug Fixes / Nits

- Fix pip install commands in LM Format Enforcer notebooks (#9648)
- Fixing some more links and documentations (#9633)
- some bedrock nits and fixes (#9646)

## [0.9.19] - 2023-12-20

### New Features

- new llama datasets `LabelledEvaluatorDataset` & `LabelledPairwiseEvaluatorDataset` (#9531)

## [0.9.18] - 2023-12-20

### New Features

- multi-doc auto-retrieval guide (#9631)

### Bug Fixes / Nits

- fix(vllm): make Vllm's 'complete' method behave the same as other LLM class (#9634)
- FIx Doc links and other documentation issue (#9632)

## [0.9.17] - 2023-12-19

### New Features

- [example] adding user feedback (#9601)
- FEATURE: Cohere ReRank Relevancy Metric for Retrieval Eval (#9495)

### Bug Fixes / Nits

- Fix Gemini Chat Mode (#9599)
- Fixed `types-protobuf` from being a primary dependency (#9595)
- Adding an optional auth token to the TextEmbeddingInference class (#9606)
- fix: out of index get latest tool call (#9608)
- fix(azure_openai.py): add missing return to subclass override (#9598)
- fix mix up b/w 'formatted' and 'format' params for ollama api call (#9594)

## [0.9.16] - 2023-12-18

### New Features

- agent refactor: step-wise execution (#9584)
- Add OpenRouter, with Mixtral demo (#9464)
- Add hybrid search to neo4j vector store (#9530)
- Add support for auth service accounts for Google Semantic Retriever (#9545)

### Bug Fixes / Nits

- Fixed missing `default=None` for `LLM.system_prompt` (#9504)
- Fix #9580 : Incorporate metadata properly (#9582)
- Integrations: Gradient[Embeddings,LLM] - sdk-upgrade (#9528)
- Add mixtral 8x7b model to anyscale available models (#9573)
- Gemini Model Checks (#9563)
- Update OpenAI fine-tuning with latest changes (#9564)
- fix/Reintroduce `WHERE` filter to the Sparse Query for PgVectorStore (#9529)
- Update Ollama API to ollama v0.1.16 (#9558)
- ollama: strip invalid `formatted` option (#9555)
- add a device in optimum push #9541 (#9554)
- Title vs content difference for Gemini Embedding (#9547)
- fix pydantic fields to float (#9542)

## [0.9.15] - 2023-12-13

### New Features

- Added full support for Google Gemini text+vision models (#9452)
- Added new Google Semantic Retriever (#9440)
- added `from_existing()` method + async support to OpenAI assistants (#9367)

### Bug Fixes / Nits

- Fixed huggingface LLM system prompt and messages to prompt (#9463)
- Fixed ollama additional kwargs usage (#9455)

## [0.9.14] - 2023-12-11

### New Features

- Add MistralAI LLM (#9444)
- Add MistralAI Embeddings (#9441)
- Add `Ollama` Embedding class (#9341)
- Add `FlagEmbeddingReranker` for reranking (#9285)
- feat: PgVectorStore support advanced metadata filtering (#9377)
- Added `sql_only` parameter to SQL query engines to avoid executing SQL (#9422)

### Bug Fixes / Nits

- Feat/PgVector Support custom hnsw.ef_search and ivfflat.probes (#9420)
- fix F1 score definition, update copyright year (#9424)
- Change more than one image input for Replicate Multi-modal models from error to warning (#9360)
- Removed GPT-Licensed `aiostream` dependency (#9403)
- Fix result of BedrockEmbedding with Cohere model (#9396)
- Only capture valid tool names in react agent (#9412)
- Fixed `top_k` being multiplied by 10 in azure cosmos (#9438)
- Fixed hybrid search for OpenSearch (#9430)

### Breaking Changes

- Updated the base `LLM` interface to match `LLMPredictor` (#9388)
- Deprecated `LLMPredictor` (#9388)

## [0.9.13] - 2023-12-06

### New Features

- Added batch prediction support for `LabelledRagDataset` (#9332)

### Bug Fixes / Nits

- Fixed save and load for faiss vector store (#9330)

## [0.9.12] - 2023-12-05

### New Features

- Added an option `reuse_client` to openai/azure to help with async timeouts. Set to `False` to see improvements (#9301)
- Added support for `vLLM` llm (#9257)
- Add support for python 3.12 (#9304)
- Support for `claude-2.1` model name (#9275)

### Bug Fixes / Nits

- Fix embedding format for bedrock cohere embeddings (#9265)
- Use `delete_kwargs` for filtering in weaviate vector store (#9300)
- Fixed automatic qdrant client construction (#9267)

## [0.9.11] - 2023-12-03

### New Features

- Make `reference_contexts` optional in `LabelledRagDataset` (#9266)
- Re-organize `download` module (#9253)
- Added document management to ingestion pipeline (#9135)
- Add docs for `LabelledRagDataset` (#9228)
- Add submission template notebook and other doc updates for `LabelledRagDataset` (#9273)

### Bug Fixes / Nits

- Convert numpy to list for `InstructorEmbedding` (#9255)

## [0.9.10] - 2023-11-30

### New Features

- Advanced Metadata filter for vector stores (#9216)
- Amazon Bedrock Embeddings New models (#9222)
- Added PromptLayer callback integration (#9190)
- Reuse file ids for `OpenAIAssistant` (#9125)

### Breaking Changes / Deprecations

- Deprecate ExactMatchFilter (#9216)

## [0.9.9] - 2023-11-29

### New Features

- Add new abstractions for `LlamaDataset`'s (#9165)
- Add metadata filtering and MMR mode support for `AstraDBVectorStore` (#9193)
- Allowing newest `scikit-learn` versions (#9213)

### Breaking Changes / Deprecations

- Added `LocalAI` demo and began deprecation cycle (#9151)
- Deprecate `QueryResponseDataset` and `DatasetGenerator` of `evaluation` module (#9165)

### Bug Fixes / Nits

- Fix bug in `download_utils.py` with pointing to wrong repo (#9215)
- Use `azure_deployment` kwarg in `AzureOpenAILLM` (#9174)
- Fix similarity score return for `AstraDBVectorStore` Integration (#9193)

## [0.9.8] - 2023-11-26

### New Features

- Add `persist` and `persist_from_dir` methods to `ObjectIndex` that are able to support it (#9064)
- Added async metadata extraction + pipeline support (#9121)
- Added back support for start/end char idx in nodes (#9143)

### Bug Fixes / Nits

- Fix for some kwargs not being set properly in global service context (#9137)
- Small fix for `memory.get()` when system/prefix messages are large (#9149)
- Minor fixes for global service context (#9137)

## [0.9.7] - 2023-11-24

### New Features

- Add support for `PGVectoRsStore` (#9087)
- Enforcing `requests>=2.31` for security, while unpinning `urllib3` (#9108)

### Bug Fixes / Nits

- Increased default memory token limit for context chat engine (#9123)
- Added system prompt to `CondensePlusContextChatEngine` that gets prepended to the `context_prompt` (#9123)
- Fixed bug in `CondensePlusContextChatEngine` not using chat history properly (#9129)

## [0.9.6] - 2023-11-22

### New Features

- Added `default_headers` argument to openai LLMs (#9090)
- Added support for `download_llama_pack()` and LlamaPack integrations
- Added support for `llamaindex-cli` command line tool

### Bug Fixed / Nits

- store normalize as bool for huggingface embedding (#9089)

## [0.9.5] - 2023-11-21

### Bug Fixes / Nits

- Fixed bug with AzureOpenAI logic for inferring if stream chunk is a tool call (#9018)

### New Features

- `FastEmbed` embeddings provider (#9043)
- More precise testing of `OpenAILike` (#9026)
- Added callback manager to each retriever (#8871)
- Ability to bypass `max_tokens` inference with `OpenAILike` (#9032)

### Bug Fixes / Nits

- Fixed bug in formatting chat prompt templates when estimating chunk sizes (#9025)
- Sandboxed Pandas execution, remediate CVE-2023-39662 (#8890)
- Restored `mypy` for Python 3.8 (#9031)
- Loosened `dataclasses-json` version range,
  and removes unnecessary `jinja2` extra from `pandas` (#9042)

## [0.9.4] - 2023-11-19

### New Features

- Added `CondensePlusContextChatEngine` (#8949)

### Smaller Features / Bug Fixes / Nits

- Fixed bug with `OpenAIAgent` inserting errors into chat history (#9000)
- Fixed various bugs with LiteLLM and the new OpenAI client (#9003)
- Added context window attribute to perplexity llm (#9012)
- Add `node_parser` attribute back to service context (#9013)
- Refactor MM retriever classes (#8998)
- Fix TextNode instantiation on SupabaseVectorIndexDemo (#8994)

## [0.9.3] - 2023-11-17

### New Features

- Add perplexity LLM integration (#8734)

### Bug Fixes / Nits

- Fix token counting for new openai client (#8981)
- Fix small pydantic bug in postgres vector db (#8962)
- Fixed `chunk_overlap` and `doc_id` bugs in `HierarchicalNodeParser` (#8983)

## [0.9.2] - 2023-11-16

### New Features

- Added new notebook guide for Multi-Modal Rag Evaluation (#8945)
- Added `MultiModalRelevancyEvaluator`, and `MultiModalFaithfulnessEvaluator` (#8945)

## [0.9.1] - 2023-11-15

### New Features

- Added Cohere Reranker fine-tuning (#8859)
- Support for custom httpx client in `AzureOpenAI` LLM (#8920)

### Bug Fixes / Nits

- Fixed issue with `set_global_service_context` not propagating settings (#8940)
- Fixed issue with building index with Google Palm embeddings (#8936)
- Fixed small issue with parsing ImageDocuments/Nodes that have no text (#8938)
- Fixed issue with large data inserts in Astra DB (#8937)
- Optimize `QueryEngineTool` for agents (#8933)

## [0.9.0] - 2023-11-15

### New Features / Breaking Changes / Deprecations

- New `IngestionPipeline` concept for ingesting and transforming data
- Data ingestion and transforms are now automatically cached
- Updated interface for node parsing/text splitting/metadata extraction modules
- Changes to the default tokenizer, as well as customizing the tokenizer
- Packaging/Installation changes with PyPi (reduced bloat, new install options)
- More predictable and consistent import paths
- Plus, in beta: MultiModal RAG Modules for handling text and images!
- Find more details at: `https://medium.com/@llama_index/719f03282945`

## [0.8.69.post1] - 2023-11-13

### Bug Fixes / Nits

- Increase max weaivate delete size to max of 10,000 (#8887)
- Final pickling remnant fix (#8902)

## [0.8.69] - 2023-11-13

### Bug Fixes / Nits

- Fixed bug in loading pickled objects (#8880)
- Fix `custom_path` vs `custom_dir` in `download_loader` (#8865)

## [0.8.68] - 2023-11-11

### New Features

- openai assistant agent + advanced retrieval cookbook (#8863)
- add retrieval API benchmark (#8850)
- Add JinaEmbedding class (#8704)

### Bug Fixes / Nits

- Improved default timeouts/retries for OpenAI (#8819)
- Add back key validation for OpenAI (#8819)
- Disable automatic LLM/Embedding model downloads, give informative error (#8819)
- fix openai assistant tool creation + retrieval notebook (#8862)
- Quick fix Replicate MultiModal example (#8861)
- fix: paths treated as hidden (#8860)
- fix Replicate multi-modal LLM + notebook (#8854)
- Feature/citation metadata (#8722)
- Fix ImageNode type from NodeWithScore for SimpleMultiModalQueryEngine (#8844)

## [0.8.67] - 2023-11-10

### New Features

- Advanced Multi Modal Retrieval Example and docs (#8822, #8823)

### Bug Fixes / Nits

- Fix retriever node postprocessors for `CitationQueryEngine` (#8818)
- Fix `cannot pickle 'builtins.CoreBPE' object` in most scenarios (#8835)

## [0.8.66] - 2023-11-09

### New Features

- Support parallel function calling with new OpenAI client in `OpenAIPydanticProgram` (#8793)

### Bug Fixes / Nits

- Fix bug in pydantic programs with new OpenAI client (#8793)
- Fixed bug with un-listable fsspec objects (#8795)

## [0.8.65] - 2023-11-08

### New Features

- `OpenAIAgent` parallel function calling (#8738)

### New Features

- Properly supporting Hugging Face recommended model (#8784)

### Bug Fixes / Nits

- Fixed missing import for `embeddings.__all__` (#8779)

### Breaking Changes / Deprecations

- Use `tool_choice` over `function_call` and `tool` over `functions` in `OpenAI(LLM)` (#8738)
- Deprecate `to_openai_function` in favor of `to_openai_tool` (#8738)

## [0.8.64] - 2023-11-06

### New Features

- `OpenAIAgent` parallel function calling (#8738)
- Add AI assistant agent (#8735)
- OpenAI GPT4v Abstraction (#8719)
- Add support for `Lantern` VectorStore (#8714)

### Bug Fixes / Nits

- Fix returning zero nodes in elastic search vector store (#8746)
- Add try/except for `SimpleDirectoryReader` loop to avoid crashing on a single document (#8744)
- Fix for `deployment_name` in async embeddings (#8748)

## [0.8.63] - 2023-11-05

### New Features

- added native sync and async client support for the lasted `openai` client package (#8712)
- added support for `AzureOpenAIEmbedding` (#8712)

### Bug Fixes / Nits

- Fixed errors about "no host supplied" with `download_loader` (#8723)

### Breaking Changes

- `OpenAIEmbedding` no longer supports azure, moved into the `AzureOpenAIEmbedding` class (#8712)

## [0.8.62.post1] - 2023-11-05

### Breaking Changes

- add new devday models (#8713)
- moved `max_docs` parameter from constructor to `lazy_load_data()` for `SimpleMongoReader` (#8686)

## [0.8.61] - 2023-11-05

### New Features

- [experimental] Hyperparameter tuner (#8687)

### Bug Fixes / Nits

- Fix typo error in CohereAIModelName class: cohere light models was missing v3 (#8684)
- Update deeplake.py (#8683)

## [0.8.60] - 2023-11-04

### New Features

- prompt optimization guide (#8659)
- VoyageEmbedding (#8634)
- Multilingual support for `YoutubeTranscriptReader` (#8673)
- emotion prompt guide (#8674)

### Bug Fixes / Nits

- Adds mistral 7b instruct v0.1 to available anyscale models (#8652)
- Make pgvector's setup (extension, schema, and table creation) optional (#8656)
- Allow init of stores_text variable for Pinecone vector store (#8633)
- fix: azure ad support (#8667)
- Fix nltk bug in multi-threaded environments (#8668)
- Fix google colab link in cohereai notebook (#8677)
- passing max_tokens to the `Cohere` llm (#8672)

## [0.8.59] - 2023-11-02

- Deepmemory support (#8625)
- Add CohereAI embeddings (#8650)
- Add Azure AD (Microsoft Entra ID) support (#8667)

## [0.8.58] - 2023-11-02

### New Features

- Add `lm-format-enforcer` integration for structured output (#8601)
- Google Vertex Support (#8626)

## [0.8.57] - 2023-10-31

### New Features

- Add `VoyageAIEmbedding` integration (#8634)
- Add fine-tuning evaluator notebooks (#8596)
- Add `SingleStoreDB` integration (#7991)
- Add support for ChromaDB PersistentClient (#8582)
- Add DataStax Astra DB support (#8609)

### Bug Fixes / Nits

- Update dataType in Weaviate (#8608)
- In Knowledge Graph Index with hybrid retriever_mode,
  - return the nodes found by keyword search when 'No Relationship found'
- Fix exceed context length error in chat engines (#8530)
- Retrieve actual content of all the triplets from KG (#8579)
- Return the nodes found by Keywords when no relationship is found by embeddings in hybrid retriever_mode in `KnowledgeGraphIndex` (#8575)
- Optimize content of retriever tool and minor bug fix (#8588)

## [0.8.56] - 2023-10-30

### New Features

- Add Amazon `BedrockEmbedding` (#8550)
- Moves `HuggingFaceEmbedding` to center on `Pooling` enum for pooling (#8467)
- Add IBM WatsonX LLM support (#8587)

### Bug Fixes / Nits

- [Bug] Patch Clarifai classes (#8529)
- fix retries for bedrock llm (#8528)
- Fix : VectorStore’s QueryResult always returns saved Node as TextNode (#8521)
- Added default file_metadata to get basic metadata that many postprocessors use, for SimpleDirectoryReader (#8486)
- Handle metadata with None values in chromadb (#8584)

## [0.8.55] - 2023-10-29

### New Features

- allow prompts to take in functions with `function_mappings` (#8548)
- add advanced prompt + "prompt engineering for RAG" notebook (#8555)
- Leverage Replicate API for serving LLaVa modal (#8539)

### Bug Fixes / Nits

- Update pull request template with google colab support inclusion (#8525)

## [0.8.54] - 2023-10-28

### New Features

- notebook showing how to fine-tune llama2 on structured outputs (#8540)
  - added GradientAIFineTuningHandler
  - added pydantic_program_mode to ServiceContext
- Initialize MultiModal Retrieval using LlamaIndex (#8507)

### Bug Fixes / Nits

- Add missing import to `ChatEngine` usage pattern `.md` doc (#8518)
- :bug: fixed async add (#8531)
- fix: add the needed CondenseQuestionChatEngine import in the usage_pa… (#8518)
- Add import LongLLMLinguaPostprocessor for LongLLMLingua.ipynb (#8519)

## [0.8.53] - 2023-10-27

### New Features

- Docs refactor (#8500)
  An overhaul of the docs organization. Major changes
  - Added a big new "understanding" section
  - Added a big new "optimizing" section
  - Overhauled Getting Started content
  - Categorized and moved module guides to a single section

## [0.8.52] - 2023-10-26

### New Features

- Add longllmlingua (#8485)
- Add google colab support for notebooks (#7560)

### Bug Fixes / Nits

- Adapt Cassandra VectorStore constructor DB connection through cassio.init (#8255)
- Allow configuration of service context and storage context in managed index (#8487)

## [0.8.51.post1] - 2023-10-25

### New Features

- Add Llava MultiModal QA examples for Tesla 10k RAG (#8271)
- fix bug streaming on react chat agent not working as expected (#8459)

### Bug Fixes / Nits

- patch: add selected result to response metadata for router query engines, fix bug (#8483)
- add Jina AI embeddings notebook + huggingface embedding fix (#8478)
- add `is_chat_model` to replicate (#8469)
- Brought back `toml-sort` to `pre-commit` (#8267)
- Added `LocationConstraint` for local `test_s3_kvstore` (#8263)

## [0.8.50] - 2023-10-24

### New Features

- Expose prompts in different modules (query engines, synthesizers, and more) (#8275)

## [0.8.49] - 2023-10-23

### New Features

- New LLM integrations
  - Support for Hugging Face Inference API's `conversational`, `text_generation`,
    and `feature_extraction` endpoints via `huggingface_hub[inference]` (#8098)
  - Add Amazon Bedrock LLMs (#8223)
  - Add AI21 Labs LLMs (#8233)
  - Add OpenAILike LLM class for OpenAI-compatible api servers (#7973)
- New / updated vector store integrations
  - Add DashVector (#7772)
  - Add Tencent VectorDB (#8173)
  - Add option for custom Postgres schema on PGVectorStore instead of only allowing public schema (#8080)
- Add Gradient fine tuning engine (#8208)
- docs(FAQ): frequently asked questions (#8249)

### Bug Fixes / Nits

- Fix inconsistencies with `ReActAgent.stream_chat` (#8147)
- Deprecate some functions for GuardrailsOutputParser (#8016)
- Simplify dependencies (#8236)
- Bug fixes for LiteLLM (#7885)
- Update for Predibase LLM (#8211)

## [0.8.48] - 2023-10-20

### New Features

- Add `DELETE` for MyScale vector store (#8159)
- Add SQL Retriever (#8197)
- add semantic kernel document format (#8226)
- Improve MyScale Hybrid Search and Add `DELETE` for MyScale vector store (#8159)

### Bug Fixes / Nits

- Fixed additional kwargs in ReActAgent.from_tools() (#8206)
- Fixed missing spaces in prompt templates (#8190)
- Remove auto-download of llama2-13B on exception (#8225)

## [0.8.47] - 2023-10-19

### New Features

- add response synthesis to text-to-SQL (#8196)
- Added support for `LLMRailsEmbedding` (#8169)
- Inferring MPS device with PyTorch (#8195)
- Consolidated query/text prepending (#8189)

## [0.8.46] - 2023-10-18

### New Features

- Add fine-tuning router support + embedding selector (#8174)
- add more document converters (#8156)

### Bug Fixes / Nits

- Add normalization to huggingface embeddings (#8145)
- Improve MyScale Hybrid Search (#8159)
- Fixed duplicate `FORMAT_STR` being inside prompt (#8171)
- Added: support for output_kwargs={'max_colwidth': xx} for PandasQueryEngine (#8110)
- Minor fix in the description for an argument in cohere llm (#8163)
- Fix Firestore client info (#8166)

## [0.8.45] - 2023-10-13

### New Features

- Added support for fine-tuning cross encoders (#7705)
- Added `QueryFusionRetriever` for merging multiple retrievers + query augmentation (#8100)
- Added `nb-clean` to `pre-commit` to minimize PR diffs (#8108)
- Support for `TextEmbeddingInference` embeddings (#8122)

### Bug Fixes / Nits

- Improved the `BM25Retriever` interface to accept `BaseNode` objects (#8096)
- Fixed bug with `BM25Retriever` tokenizer not working as expected (#8096)
- Brought mypy to pass in Python 3.8 (#8107)
- `ReActAgent` adding missing `super().__init__` call (#8125)

## [0.8.44] - 2023-10-12

### New Features

- add pgvector sql query engine (#8087)
- Added HoneyHive one-click observability (#7944)
- Add support for both SQLAlchemy V1 and V2 (#8060)

## [0.8.43.post1] - 2023-10-11

### New Features

- Moves `codespell` to `pre-commit` (#8040)
- Added `prettier` for autoformatting extensions besides `.py` (#8072)

### Bug Fixes / Nits

- Fixed forgotten f-str in `HuggingFaceLLM` (#8075)
- Relaxed numpy/panadas reqs

## [0.8.43] - 2023-10-10

### New Features

- Added support for `GradientEmbedding` embed models (#8050)

### Bug Fixes / Nits

- added `messages_to_prompt` kwarg to `HuggingFaceLLM` (#8054)
- improved selection and sql parsing for open-source models (#8054)
- fixed bug when agents hallucinate too many kwargs for a tool (#8054)
- improved prompts and debugging for selection+question generation (#8056)

## [0.8.42] - 2023-10-10

### New Features

- `LocalAI` more intuitive module-level var names (#8028)
- Enable `codespell` for markdown docs (#7972)
- add unstructured table element node parser (#8036)
- Add: Async upserting for Qdrant vector store (#7968)
- Add cohere llm (#8023)

### Bug Fixes / Nits

- Parse multi-line outputs in react agent answers (#8029)
- Add properly named kwargs to keyword `as_retriever` calls (#8011)
- Updating Reference to RAGAS LlamaIndex Integration (#8035)
- Vectara bugfix (#8032)
- Fix: ChromaVectorStore can attempt to add in excess of chromadb batch… (#8019)
- Fix get_content method in Mbox reader (#8012)
- Apply kwarg filters in WeaviateVectorStore (#8017)
- Avoid ZeroDivisionError (#8027)
- `LocalAI` intuitive module-level var names (#8028)
- zep/fix: imports & typing (#8030)
- refactor: use `str.join` (#8020)
- use proper metadata str for node parsing (#7987)

## [0.8.41] - 2023-10-07

### New Features

- You.com retriever (#8024)
- Pull fields from mongodb into metadata with `metadata_names` argument (#8001)
- Simplified `LocalAI.__init__` preserving the same behaviors (#7982)

### Bug Fixes / Nits

- Use longest metadata string for metadata aware text splitting (#7987)
- Handle lists of strings in mongodb reader (#8002)
- Removes `OpenAI.class_type` as it was dead code (#7983)
- Fixing `HuggingFaceLLM.device_map` type hint (#7989)

## [0.8.40] - 2023-10-05

### New Features

- Added support for `Clarifai` LLM (#7967)
- Add support for function fine-tuning (#7971)

### Breaking Changes

- Update document summary index (#7815)
  - change default retrieval mode to embedding
  - embed summaries into vector store by default at indexing time (instead of calculating embedding on the fly)
  - support configuring top k in llm retriever

## [0.8.39] - 2023-10-03

### New Features

- Added support for pydantic object outputs with query engines (#7893)
- `ClarifaiEmbedding` class added for embedding support (#7940)
- Markdown node parser, flat file reader and simple file node parser (#7863)
- Added support for mongdb atlas `$vectorSearch` (#7866)

### Bug Fixes / Nits

- Adds support for using message metadata in discord reader (#7906)
- Fix `LocalAI` chat capability without `max_tokens` (#7942)
- Added `codespell` for automated checking (#7941)
- `ruff` modernization and autofixes (#7889)
- Implement own SQLDatabase class (#7929)
- Update LlamaCPP context_params property (#7945)
- fix duplicate embedding (#7949)
- Adds `codespell` tool for enforcing good spelling (#7941)
- Supporting `mypy` local usage with `venv` (#7952)
- Vectara - minor update (#7954)
- Avoiding `pydantic` reinstalls in CI (#7956)
- move tree_sitter_languages into data_requirements.txt (#7955)
- Add `cache_okay` param to `PGVectorStore` to help suppress TSVector warnings (#7950)

## [0.8.38] - 2023-10-02

### New Features

- Updated `KeywordNodePostprocessor` to use spacy to support more languages (#7894)
- `LocalAI` supporting global or per-query `/chat/completions` vs `/completions` (#7921)
- Added notebook on using REBEL + Wikipedia filtering for knowledge graphs (#7919)
- Added support for `ElasticsearchEmbedding` (#7914)

## [0.8.37] - 2023-09-30

### New Features

- Supporting `LocalAI` LLMs (#7913)
- Validations protecting against misconfigured chunk sizes (#7917)

### Bug Fixes / Nits

- Simplify NL SQL response to SQL parsing, with expanded NL SQL prompt (#7868)
- Improve vector store retrieval speed for vectordb integrations (#7876)
- Added replacing {{ and }}, and fixed JSON parsing recursion (#7888)
- Nice-ified JSON decoding error (#7891)
- Nice-ified SQL error from LLM not providing SQL (#7900)
- Nice-ified `ImportError` for `HuggingFaceLLM` (#7904)
- eval fixes: fix dataset response generation, add score to evaluators (#7915)

## [0.8.36] - 2023-09-27

### New Features

- add "build RAG from scratch notebook" - OSS/local (#7864)

### Bug Fixes / Nits

- Fix elasticsearch hybrid scoring (#7852)
- Replace `get_color_mapping` and `print_text` Langchain dependency with internal implementation (#7845)
- Fix async streaming with azure (#7856)
- Avoid `NotImplementedError()` in sub question generator (#7855)
- Patch predibase initialization (#7859)
- Bumped min langchain version and changed prompt imports from langchain (#7862)

## [0.8.35] - 2023-09-27

### Bug Fixes / Nits

- Fix dropping textnodes in recursive retriever (#7840)
- share callback_manager between agent and its llm when callback_manager is None (#7844)
- fix pandas query engine (#7847)

## [0.8.34] - 2023-09-26

### New Features

- Added `Konko` LLM support (#7775)
- Add before/after context sentence (#7821)
- EverlyAI integration with LlamaIndex through OpenAI library (#7820)
- add Arize Phoenix tracer to global handlers (#7835)

### Bug Fixes / Nits

- Normalize scores returned from ElasticSearch vector store (#7792)
- Fixed `refresh_ref_docs()` bug with order of operations (#7664)
- Delay postgresql connection for `PGVectorStore` until actually needed (#7793)
- Fix KeyError in delete method of `SimpleVectorStore` related to metadata filters (#7829)
- Fix KeyError in delete method of `SimpleVectorStore` related to metadata filters (#7831)
- Addressing PyYAML import error (#7784)
- ElasticsearchStore: Update User-Agent + Add example docker compose (#7832)
- `StorageContext.persist` supporting `Path` (#7783)
- Update ollama.py (#7839)
- fix bug for self.\_session_pool (#7834)

## [0.8.33] - 2023-09-25

### New Features

- add pairwise evaluator + benchmark auto-merging retriever (#7810)

### Bug Fixes / Nits

- Minor cleanup in embedding class (#7813)
- Misc updates to `OpenAIEmbedding` (#7811)

## [0.8.32] - 2023-09-24

### New Features

- Added native support for `HuggingFaceEmbedding`, `InstructorEmbedding`, and `OptimumEmbedding` (#7795)
- Added metadata filtering and hybrid search to MyScale vector store (#7780)
- Allowing custom text field name for Milvus (#7790)
- Add support for `vector_store_query_mode` to `VectorIndexAutoRetriever` (#7797)

### Bug Fixes / Nits

- Update `LanceDBVectorStore` to handle score and distance (#7754)
- Pass LLM to `memory_cls` in `CondenseQuestionChatEngine` (#7785)

## [0.8.31] - 2023-09-22

### New Features

- add pydantic metadata extractor (#7778)
- Allow users to set the embedding dimensions in azure cognitive vector store (#7734)
- Add semantic similarity evaluator (#7770)

### Bug Fixes / Nits

- 📝docs: Update Chatbot Tutorial and Notebook (#7767)
- Fixed response synthesizers with empty nodes (#7773)
- Fix `NotImplementedError` in auto vector retriever (#7764)
- Multiple kwargs values in "KnowledgeGraphQueryEngine" bug-fix (#7763)
- Allow setting azure cognitive search dimensionality (#7734)
- Pass service context to index for dataset generator (#7748)
- Fix output parsers for selector templates (#7774)
- Update Chatbot_SEC.ipynb (#7711)
- linter/typechecker-friendly improvements to cassandra test (#7771)
- Expose debug option of `PgVectorStore` (#7776)
- llms/openai: fix Azure OpenAI by considering `prompt_filter_results` field (#7755)

## [0.8.30] - 2023-09-21

### New Features

- Add support for `gpt-3.5-turbo-instruct` (#7729)
- Add support for `TimescaleVectorStore` (#7727)
- Added `LongContextReorder` for lost-in-the-middle issues (#7719)
- Add retrieval evals (#7738)

### Bug Fixes / Nits

- Added node post-processors to async context chat engine (#7731)
- Added unique index name for postgres tsv column (#7741)

## [0.8.29.post1] - 2023-09-18

### Bug Fixes / Nits

- Fix langchain import error for embeddings (#7714)

## [0.8.29] - 2023-09-18

### New Features

- Added metadata filtering to the base simple vector store (#7564)
- add low-level router guide (#7708)
- Add CustomQueryEngine class (#7703)

### Bug Fixes / Nits

- Fix context window metadata in lite-llm (#7696)

## [0.8.28] - 2023-09-16

### New Features

- Add CorrectnessEvaluator (#7661)
- Added support for `Ollama` LLMs (#7635)
- Added `HWPReader` (#7672)
- Simplified portkey LLM interface (#7669)
- Added async operation support to `ElasticsearchStore` vector store (#7613)
- Added support for `LiteLLM` (#7600)
- Added batch evaluation runner (#7692)

### Bug Fixes / Nits

- Avoid `NotImplementedError` for async langchain embeddings (#7668)
- Imrpoved reliability of LLM selectors (#7678)
- Fixed `query_wrapper_prompt` and `system_prompt` for output parsers and completion models (#7678)
- Fixed node attribute inheritance in citation query engine (#7675)

### Breaking Changes

- Refactor and update `BaseEvaluator` interface to be more consistent (#7661)
  - Use `evaluate` function for generic input
  - Use `evaluate_response` function with `Response` objects from llama index query engine
- Update existing evaluators with more explicit naming
  - `ResponseEvaluator` -> `FaithfulnessEvaluator`
  - `QueryResponseEvaluator` -> `RelevancyEvaluator`
  - old names are kept as class aliases for backwards compatibility

## [0.8.27] - 2023-09-14

### New Features

- add low-level tutorial section (#7673)

### Bug Fixes / Nits

- default delta should be a dict (#7665)
- better query wrapper logic on LLMPredictor (#7667)

## [0.8.26] - 2023-09-12

### New Features

- add non-linear embedding adapter (#7658)
- Add "finetune + RAG" evaluation to knowledge fine-tuning notebook (#7643)

### Bug Fixes / Nits

- Fixed chunk-overlap for sentence splitter (#7590)

## [0.8.25] - 2023-09-12

### New Features

- Added `AGENT_STEP` callback event type (#7652)

### Bug Fixes / Nits

- Allowed `simple` mode to work with `as_chat_engine()` (#7637)
- Fixed index error in azure streaming (#7646)
- Removed `pdb` from llama-cpp (#7651)

## [0.8.24] - 2023-09-11

## New Features

- guide: fine-tuning to memorize knowledge (#7626)
- added ability to customize prompt template for eval modules (#7626)

### Bug Fixes

- Properly detect `llama-cpp-python` version for loading the default GGML or GGUF `llama2-chat-13b` model (#7616)
- Pass in `summary_template` properly with `RetrieverQueryEngine.from_args()` (#7621)
- Fix span types in wandb callback (#7631)

## [0.8.23] - 2023-09-09

### Bug Fixes

- Make sure context and system prompt is included in prompt for first chat for llama2 (#7597)
- Avoid negative chunk size error in refine process (#7607)
- Fix relationships for small documents in hierarchical node parser (#7611)
- Update Anyscale Endpoints integration with full streaming and async support (#7602)
- Better support of passing credentials as LLM constructor args in `OpenAI`, `AzureOpenAI`, and `Anyscale` (#7602)

### Breaking Changes

- Update milvus vector store to support filters and dynamic schemas (#7286)
  - See the [updated notebook](https://gpt-index.readthedocs.io/en/stable/examples/vector_stores/MilvusIndexDemo.html) for usage
- Added NLTK to core dependencies to support the default sentence splitter (#7606)

## [0.8.22] - 2023-09-07

### New Features

- Added support for ElasticSearch Vector Store (#7543)

### Bug Fixes / Nits

- Fixed small `_index` bug in `ElasticSearchReader` (#7570)
- Fixed bug with prompt helper settings in global service contexts (#7576)
- Remove newlines from openai embeddings again (#7588)
- Fixed small bug with setting `query_wrapper_prompt` in the service context (#7585)

### Breaking/Deprecated API Changes

- Clean up vector store interface to use `BaseNode` instead of `NodeWithEmbedding`
  - For majority of users, this is a no-op change
  - For users directly operating with the `VectorStore` abstraction and manually constructing `NodeWithEmbedding` objects, this is a minor breaking change. Use `TextNode` with `embedding` set directly, instead of `NodeWithEmbedding`.

## [0.8.21] - 2023-09-06

### New Features

- add embedding adapter fine-tuning engine + guide (#7565)
- Added support for Azure Cognitive Search vector store (#7469)
- Support delete in supabase (#6951)
- Added support for Espilla vector store (#7539)
- Added support for AnyScale LLM (#7497)

### Bug Fixes / Nits

- Default to user-configurable top-k in `VectorIndexAutoRetriever` (#7556)
- Catch validation errors for structured responses (#7523)
- Fix streaming refine template (#7561)

## [0.8.20] - 2023-09-04

### New Features

- Added Portkey LLM integration (#7508)
- Support postgres/pgvector hybrid search (#7501)
- upgrade recursive retriever node reference notebook (#7537)

## [0.8.19] - 2023-09-03

### New Features

- replace list index with summary index (#7478)
- rename list index to summary index part 2 (#7531)

## [0.8.18] - 2023-09-03

### New Features

- add agent finetuning guide (#7526)

## [0.8.17] - 2023-09-02

### New Features

- Make (some) loaders serializable (#7498)
- add node references to recursive retrieval (#7522)

### Bug Fixes / Nits

- Raise informative error when metadata is too large during splitting (#7513)
- Allow langchain splitter in simple node parser (#7517)

## [0.8.16] - 2023-09-01

### Bug Fixes / Nits

- fix link to Marvin notebook in docs (#7504)
- Ensure metadata is not `None` in `SimpleWebPageReader` (#7499)
- Fixed KGIndex visualization (#7493)
- Improved empty response in KG Index (#7493)

## [0.8.15] - 2023-08-31

### New Features

- Added support for `MarvinEntityExtractor` metadata extractor (#7438)
- Added a url_metadata callback to SimpleWebPageReader (#7445)
- Expanded callback logging events (#7472)

### Bug Fixes / Nits

- Only convert newlines to spaces for text 001 embedding models in OpenAI (#7484)
- Fix `KnowledgeGraphRagRetriever` for non-nebula indexes (#7488)
- Support defined embedding dimension in `PGVectorStore` (#7491)
- Greatly improved similarity calculation speed for the base vector store (#7494)

## [0.8.14] - 2023-08-30

### New Features

- feat: non-kg heterogeneous graph support in Graph RAG (#7459)
- rag guide (#7480)

### Bug Fixes / Nits

- Improve openai fine-tuned model parsing (#7474)
- doing some code de-duplication (#7468)
- support both str and templates for query_wrapper_prompt in HF LLMs (#7473)

## [0.8.13] - 2023-08-29

### New Features

- Add embedding finetuning (#7452)
- Added support for RunGPT LLM (#7401)
- Integration guide and notebook with DeepEval (#7425)
- Added `VectorIndex` and `VectaraRetriever` as a managed index (#7440)
- Added support for `to_tool_list` to detect and use async functions (#7282)

## [0.8.12] - 2023-08-28

### New Features

- add openai finetuning class (#7442)
- Service Context to/from dict (#7395)
- add finetuning guide (#7429)

### Smaller Features / Nits / Bug Fixes

- Add example how to run FalkorDB docker (#7441)
- Update root.md to use get_response_synthesizer expected type. (#7437)
- Bugfix MonsterAPI Pydantic version v2/v1 support. Doc Update (#7432)

## [0.8.11.post3] - 2023-08-27

### New Features

- AutoMergingRetriever (#7420)

## [0.8.10.post1] - 2023-08-25

### New Features

- Added support for `MonsterLLM` using MonsterAPI (#7343)
- Support comments fields in NebulaGraphStore and int type VID (#7402)
- Added configurable endpoint for DynamoDB (#6777)
- Add structured answer filtering for Refine response synthesizer (#7317)

### Bug Fixes / Nits

- Use `utf-8` for json file reader (#7390)
- Fix entity extractor initialization (#7407)

## [0.8.9] - 2023-08-24

### New Features

- Added support for FalkorDB/RedisGraph graph store (#7346)
- Added directed sub-graph RAG (#7378)
- Added support for `BM25Retriever` (#7342)

### Bug Fixes / Nits

- Added `max_tokens` to `Xinference` LLM (#7372)
- Support cache dir creation in multithreaded apps (#7365)
- Ensure temperature is a float for openai (#7382)
- Remove duplicate subjects in knowledge graph retriever (#7378)
- Added support for both pydantic v1 and v2 to allow other apps to move forward (#7394)

### Breaking/Deprecated API Changes

- Refactor prompt template (#7319)
  - Use `BasePromptTemplate` for generic typing
  - Use `PromptTemplate`, `ChatPromptTemplate`, `SelectorPromptTemplate` as core implementations
  - Use `LangchainPromptTemplate` for compatibility with Langchain prompt templates
  - Fully replace specific prompt classes (e.g. `SummaryPrompt`) with generic `BasePromptTemplate` for typing in codebase.
  - Keep `Prompt` as an alias for `PromptTemplate` for backwards compatibility.
  - BREAKING CHANGE: remove support for `Prompt.from_langchain_prompt`, please use `template=LangchainPromptTemplate(lc_template)` instead.

## [0.8.8] - 2023-08-23

### New Features

- `OpenAIFineTuningHandler` for collecting LLM inputs/outputs for OpenAI fine tuning (#7367)

### Bug Fixes / Nits

- Add support for `claude-instant-1.2` (#7369)

## [0.8.7] - 2023-08-22

### New Features

- Support fine-tuned OpenAI models (#7364)
- Added support for Cassandra vector store (#6784)
- Support pydantic fields in tool functions (#7348)

### Bug Fixes / Nits

- Fix infinite looping with forced function call in `OpenAIAgent` (#7363)

## [0.8.6] - 2023-08-22

### New Features

- auto vs. recursive retriever notebook (#7353)
- Reader and Vector Store for BagelDB with example notebooks (#7311)

### Bug Fixes / Nits

- Use service context for intermediate index in retry source query engine (#7341)
- temp fix for prompt helper + chat models (#7350)
- Properly skip unit-tests when packages not installed (#7351)

## [0.8.5.post2] - 2023-08-20

### New Features

- Added FireStore docstore/index store support (#7305)
- add recursive agent notebook (#7330)

### Bug Fixes / Nits

- Fix Azure pydantic error (#7329)
- fix callback trace ids (make them a context var) (#7331)

## [0.8.5.post1] - 2023-08-18

### New Features

- Awadb Vector Store (#7291)

### Bug Fixes / Nits

- Fix bug in OpenAI llm temperature type

## [0.8.5] - 2023-08-18

### New Features

- Expose a system prompt/query wrapper prompt in the service context for open-source LLMs (#6647)
- Changed default MyScale index format to `MSTG` (#7288)
- Added tracing to chat engines/agents (#7304)
- move LLM and embeddings to pydantic (#7289)

### Bug Fixes / Nits

- Fix sentence splitter bug (#7303)
- Fix sentence splitter infinite loop (#7295)

## [0.8.4] - 2023-08-17

### Bug Fixes / Nits

- Improve SQL Query parsing (#7283)
- Fix loading embed_model from global service context (#7284)
- Limit langchain version until we migrate to pydantic v2 (#7297)

## [0.8.3] - 2023-08-16

### New Features

- Added Knowledge Graph RAG Retriever (#7204)

### Bug Fixes / Nits

- accept `api_key` kwarg in OpenAI LLM class constructor (#7263)
- Fix to create separate queue instances for separate instances of `StreamingAgentChatResponse` (#7264)

## [0.8.2.post1] - 2023-08-14

### New Features

- Added support for Rockset as a vector store (#7111)

### Bug Fixes

- Fixed bug in service context definition that could disable LLM (#7261)

## [0.8.2] - 2023-08-14

### New Features

- Enable the LLM or embedding model to be disabled by setting to `None` in the service context (#7255)
- Resolve nearly any huggingface embedding model using the `embed_model="local:<model_name>"` syntax (#7255)
- Async tool-calling support (#7239)

### Bug Fixes / Nits

- Updated supabase kwargs for add and query (#7103)
- Small tweak to default prompts to allow for more general purpose queries (#7254)
- Make callback manager optional for `CustomLLM` + docs update (#7257)

## [0.8.1] - 2023-08-13

### New Features

- feat: add node_postprocessors to ContextChatEngine (#7232)
- add ensemble query engine tutorial (#7247)

### Smaller Features

- Allow EMPTY keys for Fastchat/local OpenAI API endpoints (#7224)

## [0.8.0] - 2023-08-11

### New Features

- Added "LLAMA_INDEX_CACHE_DIR" to control cached files (#7233)
- Default to pydantic selectors when possible (#7154, #7223)
- Remove the need for langchain wrappers on `embed_model` in the service context (#7157)
- Metadata extractors take an `LLM` object now, in addition to `LLMPredictor` (#7202)
- Added local mode + fallback to llama.cpp + llama2 (#7200)
- Added local fallback for embeddings to `BAAI/bge-small-en` (#7200)
- Added `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor` (#7211)

### Breaking Changes

- Change default LLM to gpt-3.5-turbo from text-davinci-003 (#7223)
- Change prompts for compact/refine/tree_summarize to work better with gpt-3.5-turbo (#7150, #7179, #7223)
- Increase default LLM temperature to 0.1 (#7180)

## [0.7.24.post1] - 2023-08-11

### Other Changes

- Reverted #7223 changes to defaults (#7235)

## [0.7.24] - 2023-08-10

### New Features

- Default to pydantic selectors when possible (#7154, #7223)
- Remove the need for langchain wrappers on `embed_model` in the service context (#7157)
- Metadata extractors take an `LLM` object now, in addition to `LLMPredictor` (#7202)
- Added local mode + fallback to llama.cpp + llama2 (#7200)
- Added local fallback for embeddings to `BAAI/bge-small-en` (#7200)
- Added `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor` (#7211)

### Breaking Changes

- Change default LLM to gpt-3.5-turbo from text-davinci-003 (#7223)
- Change prompts for compact/refine/tree_summarize to work better with gpt-3.5-turbo (#7150, #7179, #7223)
- Increase default LLM temperature to 0.1 (#7180)

### Other Changes

- docs: Improvements to Mendable Search (#7220)
- Refactor openai agent (#7077)

### Bug Fixes / Nits

- Use `1 - cosine_distance` for pgvector/postgres vector db (#7217)
- fix metadata formatting and extraction (#7216)
- fix(readers): Fix non-ASCII JSON Reader bug (#7086)
- Chore: change PgVectorStore variable name from `sim` to `distance` for clarity (#7226)

## [0.7.23] - 2023-08-10

### Bug Fixes / Nits

- Fixed metadata formatting with custom tempalates and inheritance (#7216)

## [0.7.23] - 2023-08-10

### New Features

- Add "one click observability" page to docs (#7183)
- Added Xorbits inference for local deployments (#7151)
- Added Zep vector store integration (#7203)
- feat/zep vectorstore (#7203)

### Bug Fixes / Nits

- Update the default `EntityExtractor` model (#7209)
- Make `ChatMemoryBuffer` pickleable (#7205)
- Refactored `BaseOpenAIAgent` (#7077)

## [0.7.22] - 2023-08-08

### New Features

- add ensemble retriever notebook (#7190)
- DOCS: added local llama2 notebook (#7146)

### Bug Fixes / Nits

- Fix for `AttributeError: 'OpenAIAgent' object has no attribute 'callback_manager'` by calling super constructor within `BaseOpenAIAgent`
- Remove backticks from nebula queries (#7192)

## [0.7.21] - 2023-08-07

### New Features

- Added an `EntityExtractor` for metadata extraction (#7163)

## [0.7.20] - 2023-08-06

### New Features

- add router module docs (#7171)
- add retriever router (#7166)

### New Features

- Added a `RouterRetriever` for routing queries to specific retrievers (#7166)

### Bug Fixes / Nits

- Fix for issue where having multiple concurrent streamed responses from `OpenAIAgent` would result in interleaving of tokens across each response stream. (#7164)
- fix llms callbacks issue (args[0] error) (#7165)

## [0.7.19] - 2023-08-04

### New Features

- Added metadata filtering to weaviate (#7130)
- Added token counting (and all callbacks) to agents and streaming (#7122)

## [0.7.18] - 2023-08-03

### New Features

- Added `to/from_string` and `to/from_dict` methods to memory objects (#7128)
- Include columns comments from db tables in table info for SQL queries (#7124)
- Add Neo4j support (#7122)

### Bug Fixes / Nits

- Added `Azure AD` validation support to the `AzureOpenAI` class (#7127)
- add `flush=True` when printing agent/chat engine response stream (#7129)
- Added `Azure AD` support to the `AzureOpenAI` class (#7127)
- Update LLM question generator prompt to mention JSON markdown (#7105)
- Fixed `astream_chat` in chat engines (#7139)

## [0.7.17] - 2023-08-02

### New Features

- Update `ReActAgent` to support memory modules (minor breaking change since the constructor takes `memory` instead of `chat_history`, but the main `from_tools` method remains backward compatible.) (#7116)
- Update `ReActAgent` to support streaming (#7119)
- Added Neo4j graph store and query engine integrations (#7122)
- add object streaming (#7117)

## [0.7.16] - 2023-07-30

### New Features

- Chat source nodes (#7078)

## [0.7.15] - 2023-07-29

### Bug Fixes / Nits

- anthropic api key customization (#7082)
- Fix broken link to API reference in Contributor Docs (#7080)
- Update vector store docs (#7076)
- Update comment (#7073)

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
- add agent docs (#6866)
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
- Fixes Azure gpt-35-turbo model not recognized (#6828)
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
- `TextNode` and `Document` must be instantiated with kwargs: `Document(text=text)` (#6586)
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
