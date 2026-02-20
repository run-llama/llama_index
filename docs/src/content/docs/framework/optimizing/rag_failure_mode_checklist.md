---
title: RAG Failure Mode Checklist
---

When your RAG pipeline isn't performing as expected, it can be difficult to pinpoint the root cause. This checklist covers the most common failure modes, their symptoms, and minimal fixes to get you back on track.

The first nine sections focus on single-query behavior (retrieval, chunking, embeddings, query formulation, and synthesis). The later sections highlight system-level issues that often show up only in larger or longer-running deployments.

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

- Experiment with **chunk size and overlap** (for example, larger chunks such as 1024+ tokens with 10-20% overlap)
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

- Implement a **document management** strategy with `doc_id` tracking and use `index.refresh_ref_docs()` to update changed documents
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

- Store your embedding model name alongside your index and log it in metadata or config
- Pin your embedding model version (for example, `text-embedding-ada-002` or a specific sentence-transformers version)
- If you change the embedding model, rebuild the entire index since you cannot mix embeddings
- Use `Settings.embed_model` consistently in both indexing and querying code paths

## 5. Embedding Model Mismatch (Wrong Model for the Domain)

**What happens:** The embedding model doesn't understand your domain's terminology, leading to poor semantic similarity for domain-specific queries.

**Symptoms:**

- Good results on general-knowledge questions but poor results on domain-specific ones
- Synonyms or jargon in your domain are not matched well
- Keyword search outperforms vector search on your dataset

**Fixes:**

- Try a **domain-adapted embedding model** (for example, fine-tuned models for legal, medical, or code)
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

**What happens:** The retriever searches across all documents when it should be scoped to a specific subset (for example, a certain date range, department, or document type).

**Symptoms:**

- Answers pull from wrong documents (for example, last year's report when the user asks about this year)
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
- Short queries (one or two words) return poor results
- Users need to "know the right words" to get good answers

**Fixes:**

- Add a **query transformation** step and use `HyDEQueryTransform` to generate a hypothetical answer and search with that
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

- Use a stronger LLM for synthesis (for example, GPT-4o over GPT-4o-mini) or adjust temperature slightly
- Customize the **QA prompt template** to explicitly instruct the LLM to use only the provided context
- Use `Refine` response mode to process each chunk sequentially rather than all at once
- Add a `system_prompt` reinforcing that the LLM should answer based on context only

## 10. Embedding Metric Mismatch (Cosine Score ≠ True Meaning)

**What happens:** The distance metric or normalization used for similarity does not line up with how meaning is distributed in your data. Very long or very generic chunks dominate similarity scores, while the truly relevant snippets sit lower in the ranking.

**Symptoms:**

- The top-1 result is clearly wrong, but relevant documents appear lower in the top-k list
- Similarity scores for relevant and irrelevant chunks are tightly clustered together
- Adding generic boilerplate text to documents changes retrieval behavior more than expected
- Manual inspection shows that retrieval quality is sensitive to small preprocessing changes

**Fixes:**

- Inspect the full top-k list and check whether relevant chunks appear but are ranked too low
- Normalize or trim overly long chunks so a few large nodes do not dominate similarity
- Consider a reranking stage that uses a different scoring function from the base vector store
- Evaluate retrieval with labeled queries before deploying (precision / recall on a small test set) and adjust `similarity_top_k` and thresholds based on results

## 11. Session and Cache Memory Breaks

**What happens:** Users expect the system to remember previous interactions or configuration, but the underlying indices, vector stores, or caches are stateless or keyed incorrectly. Retrieval appears flaky across sessions even though the data is present.

**Symptoms:**

- The same question asked on different days returns answers from different subsets of documents
- After a redeploy or cache clear, previously stable queries start to drift
- Manually hitting the vector store with the same query sometimes returns an empty or much smaller result set

**Fixes:**

- Define a clear strategy for **session and user keys** and ensure they are passed consistently through your application, retrievers, and stores
- Separate long-term knowledge from short-lived scratch space so cache eviction does not remove critical data
- Log index versions, cache keys, and retrieval parameters for problematic requests and compare across sessions
- Add a regression test that replays a short conversation or query sequence after each deploy to verify stability

## 12. Observability Gaps ("Black-Box Debugging")

**What happens:** You know that answers are wrong, but you cannot see what the retriever or LLM actually did. Without basic traces it becomes impossible to tell whether the issue is retrieval, synthesis, or deployment.

**Symptoms:**

- Bugs are reported as "it sometimes answers strangely" without reproducible traces
- You cannot easily inspect which nodes were retrieved for a given bad answer
- Token counts, prompts, and index metadata are not logged, so production runs cannot be reconstructed

**Fixes:**

- Enable tracing and logging for retrieval, query transforms, prompts, and responses (either through LlamaIndex callbacks or your own logging stack)
- For each failed answer, capture at least: the user query, retrieved nodes, similarity scores, index or snapshot identifiers, and the final LLM prompt
- Add a "debug mode" in your application that prints or stores retrieval results and decisions for manual inspection
- Before changing infrastructure, try to reproduce failures using only logs and traces; if you cannot, improve observability first

## 13. Index Lifecycle and Deployment Ordering

**What happens:** The pipeline works in local tests, but production behaves randomly because indices are empty, half-built, or misaligned with the running configuration. Services may start in the wrong order or point at the wrong storage.

**Symptoms:**

- Right after deployment, some queries return obviously incomplete or empty answers
- Logs show that the vector store contains far fewer nodes than expected
- Changing environment variables or secrets silently switches the index or embedding settings used at query time
- Rolling back or redeploying changes answers again without any code changes

**Fixes:**

- Treat the index as a versioned artifact and track an **index version or snapshot id** in both ingestion and serving paths
- Add a health check that runs a known-good query after deploy and fails if the index is empty, below a minimum size, or built with the wrong embedding config
- Ensure ingestion or refresh jobs complete before routing production traffic to the new index
- Avoid manual one-off ingestion steps; encode them in scripts or pipelines so they cannot be skipped accidentally

## Quick Diagnostic Flowchart

Use this to narrow down where the problem is:

1. **Check retrieval first:** Print retrieved nodes and verify they contain the answer manually.

   - If **retrieved nodes are wrong** → Focus on items 1-5, 7-8 above.
   - If **retrieved nodes are correct** → Focus on items 6, 9 above.

2. **Check a known-good query:** Try a query where you know exactly which document contains the answer.

   - If it **fails** → Likely an indexing or embedding issue (items 3-5, 10, 13).
   - If it **works** → The issue is query-specific (items 1, 7-8, 11).

3. **Check token counts:** Log the total tokens sent to the LLM.

   - If **near the limit** → Context window overflow (item 6).
   - If **well within limits** → Synthesis or retrieval quality issue (items 1-5, 8-10, 12).

4. **If problems only appear in production or after deploys:**  

   - Focus on system-level issues (items 11-13) and verify index versions, caches, and traces.

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
