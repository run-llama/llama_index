# CHANGELOG — llama-index-indices-managed-vectara

## [0.4.3]

Added llm_name argument.

## [0.4.1]

Added vectara_base_url to support custom Vectara server, and vectara_verify_ssl to enable ignoring SSL when needed.

## [0.4.0]

Implementation switched from using Vectara API v1 to API v2.
There are a number of breaking changes involved with this transition:

1. The `vectara_customer_id` parameter was removed from `VectaraIndex`. You no longer need to specify this information when you instantiate an index nor provide the environment variable `VECTARA_CUSTOMER_ID`.
2. The `vectara_corpus_id` parameter was replaced with `vectara_corpus_key`. When creating a `VectaraIndex` object, please either specify `vectara_corpus_key` explicitly or add `VECTARA_CORPUS_KEY` to your environment. This should use the corpus key of your Vectara corpus rather than the corpus ID.
3. The `add_documents()` function was removed and replaced with two new functions for indexing documents. If you want to use the Structured Document type, use the new `add_document()` function. If you would like to use the Core Document type, use the new `add_nodes()` function.
4. For specifying reranker types, `"udf"` has been replaced with `"userfn"`.
