---
title: RAG Failure Mode Checklist
---

When your RAG pipeline isn't performing as expected, it can be difficult to pinpoint the root cause. This checklist covers the most common failure modes, their symptoms, and minimal fixes to get you back on track.

## 1. Retrieval Hallucination

**What happens:** The retriever returns chunks that look superficially relevant but don't actually contain the answer. The LLM then "hallucinates" a plausible-sounding response from irrelevant context.

**Symptoms:**

- Answers sound confident but are factually wrong
- Retrieved chunks share keywords with the query but discuss a different topic
- High similarity scores on irrelevant passages

**Fixes:**

- Add a **reranker** (e.g., `CohereRerank`, `SentenceTransformerRerank`) to filter out false positives after initial retrieval
- Increase `similarity_top_k` and let the reranker prune, rather than relying on a small top-k from the vector store alone
- Use **hybrid search** (combining vector + keyword retrieval) to reduce semantic false matches
- Add a relevance threshold via `node_postprocessors` to discard low-confidence chunks

## 2. Wrong Chunk Selection (Poor Chunking)

**What happens:** Your documents are split in a way that separates critical context across multiple chunks, so no single chunk has enough information to answer the query.

**Symptoms:**

- Partial or incomplete answers
- The correct information exists in your corpus but the retrieved chunks don't contain it fully
- Answers improve dramatically when you manually provide the right text

**Fixes:**

- Experiment with **chunk size and overlap** — try larger chunks (1024+ tokens) with 10-20% overlap
- Use `SentenceSplitter` instead of naive fixed-size splitting to preserve sentence boundaries
- Try **hierarchical chunking** with `HierarchicalNodeParser` to capture both fine-grained and broad context
- Consider `SentenceWindowNodeParser` to retrieve a sentence but synthesize with surrounding context

## 3. Index Fragmentation

**What happens:** Your index contains duplicate, outdated, or conflicting versions of documents, leading to inconsistent or contradictory retrieval results.

**Symptoms:**

- Different answers for the same question across runs
- Contradictory information in retrieved chunks
- Stale data appearing even after you've updated source documents

**Fixes:**

- Implement a **document management** strategy with `doc_id` tracking — use `index.refresh_ref_docs()` to update changed documents
- Use `IngestionPipeline` with a `docstore` to deduplicate documents before indexing
- Periodically rebuild your index from source rather than only appending
- Add metadata (timestamps, version numbers) and filter on recency when relevant

## 4. Config Drift (Embedding Mismatch)

**What happens:** The embedding model used at query time differs from the one used at index time, or settings like chunk size changed between indexing and querying.

**Symptoms:**

- Sudden drop in retrieval quality after a code change or dependency update
- Similarity scores are abnormally low across all queries
- Previously working queries now return irrelevant results

**Fixes:**

- **Always store your embedding model name alongside your index** — log it in metadata or config
- Pin your embedding model version (e.g., `text-embedding-ada-002` or a specific sentence-transformers version)
- If you change the embedding model, **rebuild the entire index** — you cannot mix embeddings
- Use `Settings.embed_model` consistently in both indexing and querying code paths

## 5. Embedding Model Mismatch (Wrong Model for the Domain)

**What happens:** The embedding model doesn't understand your domain's terminology, leading to poor semantic similarity for domain-specific queries.

**Symptoms:**

- Good results on general-knowledge questions but poor results on domain-specific ones
- Synonyms or jargon in your domain aren't being matched
- Keyword search outperforms vector search on your dataset

**Fixes:**

- Try a **domain-adapted embedding model** (e.g., fine-tuned models for legal, medical, or code)
- Combine vector search with **keyword search** (hybrid mode) to capture exact terminology matches
- Generate synthetic QA pairs from your corpus and evaluate embedding recall before deploying

## 6. Context Window Overflow

**What happens:** Too many retrieved chunks are stuffed into the LLM's prompt, exceeding the context window or diluting the signal with noise.

**Symptoms:**

- Truncated or cut-off responses
- API errors about token limits
- Answers that ignore relevant retrieved content (especially content at the beginning or middle of the context)
- Degraded quality as `similarity_top_k` increases

**Fixes:**

- Use a **response synthesizer** like `TreeSummarize` or `Refine` instead of `CompactAndRefine` to handle large amounts of context
- Reduce `similarity_top_k` and rely on reranking to surface only the most relevant chunks
- Set explicit `max_tokens` limits and monitor token counts with `callback_manager`
- Consider a **summary index** or **recursive retrieval** to compress context before synthesis

## 7. Missing Metadata Filtering

**What happens:** The retriever searches across all documents when it should be scoped to a specific subset (e.g., a particular date range, department, or document type).

**Symptoms:**

- Answers pull from wrong documents (e.g., last year's report when the user asks about this year)
- Cross-contamination between document categories
- Users report "the system knows too much" or returns unrelated content

**Fixes:**

- Add structured **metadata** (date, source, category, author) during ingestion
- Use `MetadataFilters` at query time to scope retrieval
- Implement **auto-retrieval** where the LLM extracts filter parameters from the user query
- Use separate indices or namespaces for logically distinct document collections

## 8. Poor Query Understanding

**What happens:** The user's query is ambiguous, too short, or phrased differently from how information is stored in your documents.

**Symptoms:**

- Simple rephrasing of the query dramatically changes results
- Short queries (1-2 words) return poor results
- Users need to "know the right words" to get good answers

**Fixes:**

- Add a **query transformation** step — use `HyDEQueryTransform` to generate a hypothetical answer and search with that
- Use `SubQuestionQueryEngine` to break complex queries into simpler sub-queries
- Implement **query rewriting** with an LLM to expand or rephrase the query
- Add a few-shot prompt with example queries to guide users

## 9. LLM Synthesis Failures

**What happens:** The retriever gets the right chunks, but the LLM fails to synthesize a good answer from them.

**Symptoms:**

- Retrieved chunks are correct (verified manually) but the answer is still wrong
- The LLM ignores provided context and answers from its training data
- Answers are overly generic despite specific context being available

**Fixes:**

- Use a stronger LLM for synthesis (e.g., GPT-4o over GPT-4o-mini) or increase temperature slightly
- Customize the **QA prompt template** to explicitly instruct the LLM to use only the provided context
- Use `Refine` response mode to process each chunk sequentially rather than all at once
- Add `system_prompt` reinforcing that the LLM should answer based on context only

## Quick Diagnostic Flowchart

Use this to narrow down where the problem is:

1. **Check retrieval first:** Print retrieved nodes and verify they contain the answer manually.

   - If **retrieved nodes are wrong** → Focus on items 1-5, 7-8 above.
   - If **retrieved nodes are correct** → Focus on items 6, 9 above.

2. **Check a known-good query:** Try a query where you know exactly which document contains the answer.

   - If it **fails** → Likely an indexing or embedding issue (items 3-5).
   - If it **works** → The issue is query-specific (items 1, 7-8).

3. **Check token counts:** Log the total tokens sent to the LLM.
   - If **near the limit** → Context window overflow (item 6).
   - If **well within limits** → Synthesis or retrieval quality issue.

## Further Reading

- [Building Performant RAG Applications for Production](/python/framework/optimizing/production_rag/)
- [Building RAG from Scratch](/python/framework/optimizing/building_rag_from_scratch/)
- [Evaluation Guide](/python/framework/optimizing/evaluation/evaluation/)

## Key LlamaIndex Classes Referenced

- [`SentenceSplitter`](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/sentence_splitter/)
- [`HierarchicalNodeParser`](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/hierarchical/)
- [`SentenceWindowNodeParser`](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/sentence_window/)
- [`CohereRerank`](https://developers.llamaindex.ai/python/framework-api-reference/postprocessor/cohere_rerank/)
- [`SentenceTransformerRerank`](https://developers.llamaindex.ai/python/framework-api-reference/postprocessor/sentence_transformer_rerank/)
- [`HyDEQueryTransform`](https://developers.llamaindex.ai/python/framework-api-reference/query/query_transform/hyde/)
- [`SubQuestionQueryEngine`](https://developers.llamaindex.ai/python/framework-api-reference/query/sub_question/)
- [`IngestionPipeline`](https://developers.llamaindex.ai/python/framework-api-reference/ingestion/pipeline/)
- [`MetadataFilters`](https://developers.llamaindex.ai/python/framework-api-reference/retrievers/vector_store/)
