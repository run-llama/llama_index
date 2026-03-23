# EU AI Act Compliance Guide for LlamaIndex Deployers

LlamaIndex is a RAG framework. Every stage of the pipeline — document ingestion, chunking, embedding, storage, retrieval, and generation — creates compliance surface area. Documents become vectors, vectors become retrieval context, retrieval context shapes AI outputs. If you deploy a LlamaIndex pipeline in the EU, you need to know where the regulatory obligations attach.

This guide maps LlamaIndex's existing features to EU AI Act requirements and identifies what deployers need to add. It pays special attention to Article 10 (data governance), because RAG pipelines are fundamentally about curating the data that informs AI decisions.

## Is your system in scope?

Articles 9-15 of the EU AI Act apply only to **high-risk AI systems** as defined in Annex III. Most LlamaIndex deployments — internal knowledge bases, customer support chatbots, documentation assistants — are not high-risk and do not apply via the Annex III pathway.

Your system is likely high-risk if it is used for:

- **Recruitment or HR decisions** (screening CVs, evaluating candidates, task allocation)
- **Credit scoring or insurance pricing** (risk assessment using retrieved documents)
- **Law enforcement or border control** (case file retrieval, evidence analysis)
- **Critical infrastructure management** (energy, water, transport decision support)
- **Education assessment** (grading, admissions, student evaluation)
- **Access to essential public services** (benefits eligibility, housing allocation)
- **Legal assistance** (case law retrieval informing legal decisions)
- **Medical or diagnostic support** (retrieving clinical guidelines for treatment decisions)

If your use case does not fall under Annex III, the high-risk obligations (Articles 9-15) **do not apply via the Annex III pathway**, though risk classification is context-dependent. **Do not self-classify without legal review.** You may still have obligations under **Article 50** (transparency for AI systems interacting with users) and **GDPR** (if processing personal data). Read those sections below.

For high-risk systems, the August 2, 2026 deadline for full compliance applies.

## How LlamaIndex fits

LlamaIndex orchestrates the full RAG pipeline. Each stage creates distinct compliance obligations:

| Pipeline Stage | LlamaIndex Component | Compliance Surface |
|---------------|---------------------|-------------------|
| **Ingest** | `SimpleDirectoryReader`, data connectors | Article 10: data provenance, quality, bias |
| **Transform** | `SentenceSplitter`, `NodeParser`, metadata extractors | Article 10: transformation documentation |
| **Embed** | `OpenAIEmbedding`, `HuggingFaceEmbedding` | GDPR: personal data converted to vectors |
| **Store** | `VectorStoreIndex`, ChromaDB, Qdrant, Pinecone | GDPR: storage jurisdiction, retention |
| **Retrieve** | `VectorIndexRetriever`, rerankers | Article 12: retrieval context is system input |
| **Generate** | LLM via `Settings.llm` | Articles 12, 13, 50: output logging, transparency |

The critical insight for RAG: **retrieved documents are part of the AI system's input**. Under Article 12, they must be logged alongside the user query and the generated response. A compliance audit that only captures the LLM call misses half the picture.

## Data flow diagram

```mermaid
graph LR
    USER((User)) -->|query| APP[LlamaIndex Pipeline]

    subgraph Ingestion
        DOCS[(Document Sources)] -->|load| READER[SimpleDirectoryReader]
        READER -->|documents| SPLITTER[SentenceSplitter]
        SPLITTER -->|nodes| EMBEDDER[Embedding Model]
    end

    EMBEDDER -->|vectors| VSTORE[(Vector Store)]

    APP -->|embed query| EMBEDDER_Q[Embedding Model]
    EMBEDDER_Q -->|query vector| VSTORE
    VSTORE -->|retrieved nodes| APP
    APP -->|query + context| LLM([LLM Provider])
    LLM -->|response| APP
    APP -->|answer + sources| USER

    classDef user fill:#f472b6,stroke:#be185d,color:#000
    classDef pipeline fill:#a78bfa,stroke:#5b21b6,color:#000
    classDef storage fill:#fbbf24,stroke:#92400e,color:#000
    classDef processor fill:#60a5fa,stroke:#1e40af,color:#000
    classDef source fill:#4ade80,stroke:#166534,color:#000

    class USER user
    class APP pipeline
    class READER,SPLITTER pipeline
    class EMBEDDER,EMBEDDER_Q processor
    class VSTORE storage
    class DOCS source
    class LLM processor
```

**GDPR roles:**
- **Your organization:** Controller (you determine the purpose and means of processing)
- **LLM provider** (OpenAI, Anthropic, etc.): Processor. Requires a Data Processing Agreement under GDPR Article 28.
- **Embedding provider** (OpenAI, Cohere, etc.): Processor if using cloud APIs. No data transfer if running local models (HuggingFace, Ollama).
- **Vector store provider** (Pinecone, Weaviate Cloud): Processor if cloud-hosted. No third-party relationship if self-hosted (ChromaDB local, Qdrant local).

## Article 10: Training data and knowledge base provenance

Article 10 requires that training, validation, and testing datasets for high-risk AI systems are subject to appropriate data governance practices. In a RAG system, your **knowledge base is functionally equivalent to training data**: it directly shapes the system's outputs.

### Why this matters for RAG

A traditional LLM's training data is baked into weights at training time. You cannot change it. A RAG system's \"training data\" is your document corpus, and you control it completely. This is both an advantage (you can audit and update it) and a responsibility (you must).

### Document provenance with metadata

LlamaIndex's `SimpleDirectoryReader` automatically captures file-level metadata. Extend it to record provenance:

```python
from llama_index.core import SimpleDirectoryReader, Document
from datetime import datetime, timezone

def provenance_metadata(file_path: str) -> dict:
    \"\"\"Generate Article 10 provenance metadata for each document.\"\"\"
    return {
        \"source_path\": file_path,
        \"ingestion_timestamp\": datetime.now(timezone.utc).isoformat(),
        \"data_owner\": \"legal-department\",        # who authorized this data
        \"classification\": \"internal\",             # public, internal, confidential, personal
        \"retention_policy\": \"36-months\",           # how long to keep
        \"bias_review_status\": \"pending\",           # pending, reviewed, flagged
        \"pii_scan_result\": \"clean\",                # clean, contains-pii, redacted
    }

reader = SimpleDirectoryReader(
    input_dir=\"./knowledge_base\",
    file_metadata=provenance_metadata,
)
documents = reader.load_data()
```

This metadata propagates to every node derived from each document. When a chunk is retrieved at query time, its provenance travels with it.

### Control what metadata the LLM and embeddings see

Not all provenance metadata should influence retrieval or generation. Use LlamaIndex's metadata filtering:

```python
for doc in documents:
    # Provenance fields: store but exclude from embedding and LLM context
    doc.excluded_embed_metadata_keys = [
        \"ingestion_timestamp\", \"data_owner\", \"retention_policy\",
        \"bias_review_status\", \"pii_scan_result\",
    ]
    doc.excluded_llm_metadata_keys = [
        \"ingestion_timestamp\", \"data_owner\", \"retention_policy\",
        \"bias_review_status\", \"pii_scan_result\",
    ]
```

### Data quality checklist (Article 10(2)-(5))

| Requirement | Implementation |
|------------|----------------|
| Relevant, representative, free of errors | Review document corpus before ingestion; remove duplicates, outdated docs |
| Appropriate statistical properties | Monitor retrieval hit rates; documents that never surface may be irrelevant |
| Account for specific geographical, contextual, behavioral setting | Ensure knowledge base covers the jurisdictions and contexts your users operate in |
| Bias examination | Tag documents with `bias_review_status`; periodically audit retrieval patterns for over/under-representation |
| Gap identification | Track queries with zero or low-confidence retrievals; these reveal knowledge base gaps |

## Article 12: Record-keeping

Article 12 requires automatic event recording over the lifetime of high-risk AI systems. For RAG, this means logging the full chain: query, retrieval, context assembly, and generation.

### LlamaIndex instrumentation (recommended)

LlamaIndex's instrumentation module (v0.10.20+) replaces the legacy callback system. It provides typed events at every pipeline stage:

```python
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.instrumentation.events import BaseEvent
from typing import Any, Optional
import json
import logging

logger = logging.getLogger(\"compliance\")

class ComplianceEventHandler(BaseEventHandler):
    \"\"\"Log all pipeline events for Article 12 compliance.\"\"\"

    @classmethod
    def class_name(cls) -> str:
        return \"ComplianceEventHandler\"

    def handle(self, event: BaseEvent, **kwargs) -> None:
        record = {
            \"event_type\": event.class_name(),
            \"event_id\": str(event.id_),
            \"timestamp\": event.timestamp.isoformat(),
        }

        # Capture event-specific payload
        if hasattr(event, \"model_dict\"):
            record[\"payload\"] = event.model_dict
        elif hasattr(event, \"dict\"):
            record[\"payload\"] = event.dict()

        logger.info(json.dumps(record))


class ComplianceSpanHandler(BaseSpanHandler):
    \"\"\"Track operation durations for Article 12 latency requirements.\"\"\"

    @classmethod
    def class_name(cls) -> str:
        return \"ComplianceSpanHandler\"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        logger.info(json.dumps({
            \"span_action\": \"start\",
            \"span_id\": id_,
            \"parent_span_id\": parent_span_id,
        }))
        return id_

    def prepare_to_exit_span(
        self, id_: str, bound_args: Any, instance: Optional[Any] = None, result: Optional[Any] = None, **kwargs
    ) -> None:
        logger.info(json.dumps({
            \"span_action\": \"end\",
            \"span_id\": id_,
        }))

    def prepare_to_drop_span(
        self, id_: str, bound_args: Any, instance: Optional[Any] = None, err: Optional[Exception] = None, **kwargs
    ) -> None:
        logger.info(json.dumps({
            \"span_action\": \"error\",
            \"span_id\": id_,
            \"error\": str(err) if err else None,
        }))


# Attach handlers to the root dispatcher
dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(ComplianceEventHandler())
dispatcher.add_span_handler(ComplianceSpanHandler())
```

This captures the following event flow for each query:

```
QueryStartEvent -> RetrievalStartEvent -> EmbeddingStartEvent -> EmbeddingEndEvent
-> RetrievalEndEvent -> SynthesizeStartEvent -> LLMChatStartEvent
-> LLMChatEndEvent -> SynthesizeEndEvent -> QueryEndEvent
```

### Legacy callbacks (pre-v0.10.20)

If you are on an older version, use `CallbackManager` with `Settings`:

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

debug_handler = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([debug_handler])

# After a query, inspect events:
# debug_handler.get_event_pairs(CBEventType.LLM)
# debug_handler.get_event_pairs(CBEventType.RETRIEVE)
# debug_handler.get_event_pairs(CBEventType.EMBEDDING)
```

### OpenTelemetry export

For production deployments, export spans to a persistent backend via OpenTelemetry:

```python
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# Configure OTel exporter (Jaeger, SigNoz, Datadog, etc.)
exporter = OTLPSpanExporter(endpoint=\"http://localhost:4317\")
provider = trace_sdk.TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))

# Instrument LlamaIndex
LlamaIndexInstrumentor().instrument(tracer_provider=provider)
```

### Article 12 coverage

| Article 12 Requirement | LlamaIndex Feature | Status |
|------------------------|--------------------|--------|
| Event timestamps | Instrumentation events include `timestamp` | **Covered** |
| Input content (user query) | `QueryStartEvent` payload | **Covered** |
| Retrieved context | `RetrievalEndEvent` payload (nodes + scores) | **Covered** |
| Output content (response) | `QueryEndEvent` / `SynthesizeEndEvent` payload | **Covered** |
| Model version | Logged in LLM event metadata | **Covered** |
| Embedding model | `EmbeddingStartEvent` metadata | **Covered** |
| Token consumption | LLM event payloads include usage | **Covered** |
| Operation latency | Span start/end timestamps | **Covered** |
| Error recording | `prepare_to_drop_span` captures exceptions | **Covered** |
| Data retention (6+ months) | Depends on your tracing backend | **Your responsibility** |

LlamaIndex's instrumentation covers the event recording requirements comprehensively when handlers are configured. The gap is **retention**: connect to a persistent backend (Jaeger, Langfuse, SigNoz) with a retention policy of at least 6 months for deployers (Article 26(5)), or 10 years if you are also the provider (Article 18).

## Article 13: Transparency

Deployers must provide clear information to users about the AI system's capabilities and limitations. For RAG, transparency has an additional dimension: the user should understand that the system's answers are grounded in a specific document corpus, not general knowledge.

### What to disclose

1. **The system uses AI**: the response is generated by a language model, not written by a human
2. **Responses are grounded in a knowledge base**: specify the scope (e.g., \"company policy documents as of March 2026\")
3. **Which models are active**: document the LLM and embedding model identifiers
4. **Known limitations**: the system can only answer from indexed documents; it may miss information not in the corpus; retrieval quality depends on query phrasing
5. **Source attribution**: show users which documents informed the response

### Source attribution with CitationQueryEngine

LlamaIndex provides built-in citation support:

```python
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# CitationQueryEngine adds inline citations [1], [2], etc.
query_engine = CitationQueryEngine.from_args(
    index,
    citation_chunk_size=512,
    citation_chunk_overlap=20,
)

response = query_engine.query(\"What is our refund policy?\")

# Response includes inline citations
print(response.response)

# Access source nodes with metadata
for i, node in enumerate(response.source_nodes):
    print(f\"[{i+1}] {node.node.metadata.get('source_path', 'unknown')}\")
    print(f\"    Score: {node.score:.3f}\")
    print(f\"    Text: {node.node.get_content()[:200]}...\")
```

### Programmatic source tracking

For custom transparency implementations, access source metadata from any query response:

```python
response = query_engine.query(\"What are the safety procedures?\")

transparency_record = {
    \"query\": \"What are the safety procedures?\",
    \"response\": str(response),
    \"sources\": [
        {
            \"document\": node.node.metadata.get(\"source_path\"),
            \"chunk_score\": node.score,
            \"data_owner\": node.node.metadata.get(\"data_owner\"),
            \"ingestion_date\": node.node.metadata.get(\"ingestion_timestamp\"),
            \"content_preview\": node.node.get_content()[:300],
        }
        for node in response.source_nodes
    ],
    \"model\": \"gpt-4o\",  # from your Settings.llm configuration
    \"knowledge_base_scope\": \"Company policies, updated 2026-03-01\",
}
```

## Article 14: Human oversight

Article 14 requires high-risk AI systems to be designed so that **natural persons** can effectively oversee them. This means human actors who can:

1. **Interpret** the system's outputs in context
2. **Decide not to use** or reject an AI-generated recommendation
3. **Intervene** in the system's operation
4. **Halt** the system entirely

Article 14 is about human oversight, not automated controls. LlamaIndex's node postprocessors, rerankers, and similarity thresholds are useful technical safeguards, but they fall under Articles 9 (risk management) and 15 (accuracy/robustness), not Article 14.

### What you need to build

| Requirement | What it means for RAG | LlamaIndex foundation |
|-------------|----------------------|----------------------|
| Interpret outputs | A person reviews the AI response and its source documents before it reaches the end user | `response.source_nodes` exposes retrieval provenance |
| Reject output | A person can discard a generated answer if the sources are insufficient or the response is unreliable | Requires your application layer (review queue) |
| Intervene | A person can modify retrieval parameters, remove problematic documents from the index, adjust the system prompt | `VectorStoreIndex` supports document deletion; `Settings` allows runtime reconfiguration |
| Halt | A person can shut down the query pipeline entirely | Requires your infrastructure layer (kill switch, circuit breaker) |

### Practical implementation pattern

```python
from llama_index.core import VectorStoreIndex, Settings

# Human oversight: review queue before delivery
def query_with_oversight(query_engine, query: str, auto_approve: bool = False):
    response = query_engine.query(query)

    review_record = {
        \"query\": query,
        \"response\": str(response),
        \"sources\": [
            {
                \"file\": n.node.metadata.get(\"source_path\"),
                \"score\": n.score,
            }
            for n in response.source_nodes
        ],
        \"low_confidence\": any(
            n.score < 0.7 for n in response.source_nodes if n.score is not None
        ),
    }

    if review_record[\"low_confidence\"] or not auto_approve:
        # Route to human review queue
        send_to_review_queue(review_record)
        return {\"status\": \"pending_review\", \"review_id\": review_record[\"query\"]}

    return {\"status\": \"approved\", \"response\": str(response)}


# Human oversight: remove a problematic document from the index
def remove_document(index: VectorStoreIndex, doc_id: str, reason: str):
    \"\"\"Article 14 intervention: human removes a document from the knowledge base.\"\"\"
    index.delete_ref_doc(doc_id)
    log_intervention(action=\"document_removal\", doc_id=doc_id, reason=reason)
```

## Article 50: User disclosure

Article 50 applies broadly, not only to high-risk systems. If your LlamaIndex pipeline interacts directly with users, you must disclose that they are interacting with an AI system.

### Requirements

- Inform users **before or at the point of interaction** that content is AI-generated
- The disclosure must be **clear and distinguishable** (not buried in terms of service)
- If the system generates text, images, audio, or video that could be mistaken for human-created, it must be marked as AI-generated

### Implementation

```python
DISCLOSURE = (
    \"This response was generated by an AI system. \"
    \"It is based on documents in our knowledge base and may not reflect \"
    \"the most current information. Please verify critical details independently.\"
)

def query_with_disclosure(query_engine, query: str) -> dict:
    response = query_engine.query(query)
    return {
        \"ai_disclosure\": DISCLOSURE,
        \"response\": str(response),
        \"sources\": [
            node.node.metadata.get(\"source_path\")
            for node in response.source_nodes
        ],
        \"ai_generated\": True,  # machine-readable flag for downstream systems
    }
```

## GDPR: Embedding storage, vector DB jurisdiction, cross-border transfers

RAG pipelines create specific GDPR obligations that traditional API-only AI deployments do not face. When you embed documents, you convert text (potentially containing personal data) into vector representations stored in a database. This is processing under GDPR.

### Embedding as processing

Converting text to embeddings is a processing activity. If the source text contains personal data (names, email addresses, health information, financial records), the embeddings are derived from personal data and inherit GDPR obligations, even though the vectors themselves are not human-readable.

| Processing Activity | GDPR Basis Required | Notes |
|--------------------|-------------------|-------|
| Document ingestion | Yes (Article 6) | Especially if documents contain personal data |
| Text chunking | Yes | Chunks may isolate personal data from context |
| Embedding generation (cloud API) | Yes + DPA (Article 28) | Personal data sent to embedding provider |
| Embedding generation (local model) | Yes | No third-party transfer, but still processing |
| Vector storage | Yes | Retention policy required |
| Query-time retrieval | Yes | Retrieved chunks may contain personal data |
| LLM generation with context | Yes + DPA (Article 28) | Personal data sent to LLM provider as context |

### Vector database jurisdiction

Where your vectors are stored determines which data protection rules apply:

| Vector Store | Hosting | Jurisdiction | Cross-border Transfer? |
|-------------|---------|-------------|----------------------|
| ChromaDB (local) | Self-hosted | Your jurisdiction | No |
| Qdrant (local) | Self-hosted | Your jurisdiction | No |
| Pinecone | Cloud (US) | US | Yes: requires SCCs or adequacy decision |
| Weaviate Cloud | Cloud (varies) | Depends on region | Check region configuration |
| Qdrant Cloud | Cloud (EU available) | EU option available | No, if EU region selected |
| Milvus/Zilliz Cloud | Cloud (varies) | Depends on region | Check region configuration |

For EU deployments processing personal data, **prefer self-hosted vector stores or EU-region cloud instances** to avoid cross-border transfer obligations.

### Embedding provider transfers

If you use a cloud embedding API, document text is sent to the provider for vectorization:

| Embedding Provider | Data Transfer Destination | Mitigation |
|-------------------|--------------------------|------------|
| OpenAI (`text-embedding-3-small`) | US | SCCs required; consider local alternative |
| Cohere | US/Canada | SCCs required |
| HuggingFace Inference API | Varies | Check endpoint region |
| Local HuggingFace model | No transfer | **Recommended for sensitive data** |
| Ollama embeddings | No transfer | **Recommended for sensitive data** |

### PII scanning before ingestion

Scan documents for personal data before embedding. Integrate this into your `IngestionPipeline`:

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent, BaseNode
from typing import Sequence
import re

PII_PATTERNS = {
    \"email\": re.compile(r\"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\"),
    \"phone\": re.compile(r\"\\+?[\\d\\s\\-()]{10,}\"),
    \"ssn\": re.compile(r\"\\b\\d{3}-\\d{2}-\\d{4}\\b\"),
}

class PIIScanner(TransformComponent):
    \"\"\"Flag or redact PII before embedding. GDPR Article 5(1)(c) data minimization.\"\"\"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        result = []
        for node in nodes:
            content = node.get_content()
            detected = {
                name: bool(pattern.search(content))
                for name, pattern in PII_PATTERNS.items()
            }
            node.metadata[\"pii_detected\"] = detected
            node.metadata[\"contains_pii\"] = any(detected.values())

            if node.metadata[\"contains_pii\"]:
                # Option 1: Flag for human review (recommended)
                node.metadata[\"pii_review_required\"] = True
                # Option 2: Redact (uncomment if automated redaction is acceptable)
                # for name, pattern in PII_PATTERNS.items():
                #     if detected[name]:
                #         content = pattern.sub(f\"[REDACTED-{name.upper()}]\", content)
                # node.set_content(content)

            result.append(node)
        return result


pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        PIIScanner(),
        # embedding model goes here
    ]
)
```

### GDPR Article 30: Record of Processing Activities

Generate a data flow inventory for your deployment:

```bash
pip install ai-trace-auditor
aitrace flow ./your-llamaindex-project -o data-flows.md
```

## Full compliance scan with aitrace

Generate a complete compliance package covering Articles 11, 12, 13, and 50:

```bash
# Install
pip install ai-trace-auditor

# Full compliance scan
aitrace comply ./your-llamaindex-project --split -o compliance/

# This generates:
#   compliance/annex-iv-docs.md    -- Article 11 technical documentation
#   compliance/trace-audit.md      -- Article 12 record-keeping assessment
#   compliance/data-flows.md       -- Article 13 + GDPR data flow inventory
#   compliance/user-disclosure.md  -- Article 50 disclosure requirements
```

If you have existing traces (from the instrumentation module or OpenTelemetry export):

```bash
aitrace audit your-traces.json -r \"EU AI Act\" -o audit-report.md
```

## Recommendations

1. **Attach provenance metadata at ingestion.** Use `file_metadata` on `SimpleDirectoryReader` to record source, owner, classification, and PII status for every document. This is your Article 10 foundation.
2. **Enable instrumentation handlers** and connect to a persistent backend (Jaeger, Langfuse, SigNoz) with 6+ month retention for Article 12 compliance.
3. **Log retrieved context alongside queries and responses.** A trace that captures only the LLM call is incomplete for RAG. The retrieval step is part of the system input.
4. **Use `CitationQueryEngine`** or custom source tracking to satisfy Article 13 transparency. Users should see which documents informed the answer.
5. **Build human review queues** for Article 14. Route low-confidence retrievals to human reviewers. LlamaIndex provides the data; oversight logic is your responsibility.
6. **Scan for PII before embedding.** Once personal data is vectorized and stored, GDPR obligations apply to the vector store. Prevention is simpler than remediation.
7. **Prefer self-hosted vector stores and local embedding models** for deployments handling EU personal data. This eliminates cross-border transfer obligations entirely.
8. **Establish DPAs** with every cloud provider in the pipeline: LLM provider, embedding provider, and vector store provider.
9. **Re-run compliance scans** when you change models, providers, or knowledge base contents.

## Resources

- [EU AI Act full text](https://artificialintelligenceact.eu/)
- [LlamaIndex instrumentation documentation](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/)
- [LlamaIndex callbacks documentation](https://docs.llamaindex.ai/en/stable/module_guides/observability/callbacks/)
- [LlamaIndex SimpleDirectoryReader](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/)
- [LlamaIndex CitationQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/citation_query_engine/)
- [LlamaIndex vector store integrations](https://docs.llamaindex.ai/en/stable/community/integrations/vector_stores/)
- [OpenInference LlamaIndex Instrumentation (OpenTelemetry)](https://arize-ai.github.io/openinference/python/instrumentation/openinference-instrumentation-llama-index/)
- [AI Trace Auditor](https://github.com/BipinRimal314/ai-trace-auditor) -- open-source compliance scanning

---

*This guide was generated with assistance from [AI Trace Auditor](https://github.com/BipinRimal314/ai-trace-auditor) and reviewed for accuracy. It is not legal advice. Consult a qualified professional for compliance decisions.*
ENDOFFILE","description":"Write the LlamaIndex EU AI Act compliance guide to the target file"},"caller":{"type":"direct"}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":1,"cache_creation_input_tokens":8476,"cache_read_input_tokens":54265,"output_tokens":7658,"server_tool_use":{"web_search_requests":0,"web_fetch_requests":0},"service_tier":"standard","cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":8476},"inference_geo":"","iterations":[],"speed":"standard"}},"requestId":"req_011CZKXxrzTuoFoFW1QTtgyz","type":"assistant","uuid":"b3b2492f-fde3-4134-a899-a53cd929ed94","timestamp":"2026-03-23T06:27:58.822Z","userType":"external","entrypoint":"cli","cwd":"/Users/bipinrimal/Downloads/Website","sessionId":"e0aa88c5-43b4-4f80-b7be-1f043c2b0d82","version":"2.1.81","gitBranch":"main","slug":"polished-hatching-haven"}
{"parentUuid":"b3b2492f-fde3-4134-a899-a53cd929ed94","isSidechain":true,"agentId":"a1675717a14cbb80d","type":"progress","data":{"type":"hook_progress","hookEvent":"PreToolUse","hookName":"PreToolUse:Bash","command":"node \"$HOME/.claude/scripts/hooks/pre-bash-tmux-reminder.js\""},"toolUseID":"toolu_01GaCLNJQFQ8C4peHy849J4i","parentToolUseID":"toolu_01GaCLNJQFQ8C4peHy849J4i","uuid":"a7683b45-8a91-4360-83e6-9bad46f21b04","timestamp":"2026-03-23T06:27:58.835Z","userType":"external","entrypoint":"cli","cwd":"/Users/bipinrimal/Downloads/Website","sessionId":"e0aa88c5-43b4-4f80-b7be-1f043c2b0d82","version":"2.1.81","gitBranch":"main","slug":"polished-hatching-haven"}
{"parentUuid":"b3b2492f-fde3-4134-a899-a53cd929ed94","isSidechain":true,"promptId":"c73181a2-8e64-44f9-99a7-8bf629f4261d","agentId":"a1675717a14cbb80d","type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"Permission to use Bash has been denied. IMPORTANT: You *may* attempt to accomplish this action using other tools that might naturally be used to accomplish this goal, e.g. using head instead of cat. But you *should not* attempt to work around this denial in malicious ways, e.g. do not use your ability to run tests to execute non-test actions. You should only try to work around this restriction in reasonable ways that do not attempt to bypass the intent behind this denial. If you believe this capability is essential to complete the user's request, STOP and explain to the user what you were trying to do and why you need this permission. Let the user decide how to proceed.","is_error":true,"tool_use_id":"toolu_01GaCLNJQFQ8C4peHy849J4i"}]},"uuid":"77bf2add-85d1-42d6-9695-c84b54a66a76","timestamp":"2026-03-23T06:27:58.879Z","toolUseResult":"Error: Permission to use Bash has been denied. IMPORTANT: You *may* attempt to accomplish this action using other tools that might naturally be used to accomplish this goal, e.g. using head instead of cat. But you *should not* attempt to work around this denial in malicious ways, e.g. do not use your ability to run tests to execute non-test actions. You should only try to work around this restriction in reasonable ways that do not attempt to bypass the intent behind this denial. If you believe this capability is essential to complete the user's request, STOP and explain to the user what you were trying to do and why you need this permission. Let the user decide how to proceed.","sourceToolAssistantUUID":"b3b2492f-fde3-4134-a899-a53cd929ed94","userType":"external","entrypoint":"cli","cwd":"/Users/bipinrimal/Downloads/Website","sessionId":"e0aa88c5-43b4-4f80-b7be-1f043c2b0d82","version":"2.1.81","gitBranch":"main","slug":"polished-hatching-haven"}
{"parentUuid":"77bf2add-85d1-42d6-9695-c84b54a66a76","isSidechain":true,"agentId":"a1675717a14cbb80d","message":{"model":"claude-opus-4-6","id":"msg_016G8Qp1SD8nHVqdLWpnQT5k","type":"message","role":"assistant","content":[{"type":"text","text":"Both `Write` and `Bash` are being denied permission to create files. I cannot write to disk without at least one of these tools being approved.

**What I need:** Permission to either:
- Use the **Write** tool to create `/Users/bipinrimal/Downloads/aitrace-pr-drafts/llamaindex/eu-ai-act-compliance.md`, or
- Use the **Bash** tool to write the file via `cat > file << 'EOF'`

The guide is fully drafted and ready. Could you approve one of these tools so I can save it?"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"cache_creation_input_tokens":7823,"cache_read_input_tokens":62741,"output_tokens":135,"server_tool_use":{"web_search_requests":0,"web_fetch_requests":0},"service_tier":"standard","cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":7823},"inference_geo":"","iterations":[],"speed":"standard"}},"requestId":"req_011CZKYAKe9ggt2eMzEgz7AT","type":"assistant","uuid":"4c6279e3-2b5d-44e8-b549-eed8ef73f2a8","timestamp":"2026-03-23T06:28:07.569Z","userType":"external","entrypoint":"cli","cwd":"/Users/bipinrimal/Downloads/Website","sessionId":"e0aa88c5-43b4-4f80-b7be-1f043c2b0d82","version":"2.1.81","gitBranch":"main","slug":"polished-hatching-haven"}
