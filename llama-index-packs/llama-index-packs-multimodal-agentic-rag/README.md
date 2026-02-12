# Multimodal Agentic RAG Pack

A high-fidelity **Multimodal Agentic RAG** LlamaPack built for complex, visually rich documents such as textbooks, research papers, and technical manuals.

Traditional RAG pipelines operate primarily on plain text. This pack goes further — it preserves **visual grounding** (images) and injects structured visual metadata into retrieval results, enabling precise frontend visualization and diagram-aware reasoning.

---

## 🚀 What This Pack Solves

Many real-world documents rely heavily on visual information:

- System architecture diagrams
- Mathematical formulas
- Scientific plots
- Structured tables

Standard text-only RAG systems lose this critical context.

**MultimodalAgenticRAGPack** ensures that:

- Visual elements are semantically understood via VLMs
- Bounding boxes are preserved for UI highlighting
- Vector and graph reasoning work together
- Retrieval is evaluated and improved via an agentic workflow

---

## ✨ Core Capabilities

### 1️⃣ Multimodal Ingestion (Hybrid Pipeline)

The ingestion process employs a hybrid strategy to ensure high-fidelity document understanding:

- **Text Extraction**: Uses high-precision PDF parsers to extract raw text and markdown tables.
- **VLM-Powered Visual Analysis**: Specifically targets images and visual components using **Qwen-VL** to extract:
  - **Dense Image Captions**: Detailed semantic descriptions of diagrams and photos.
  - **Structured Visual Descriptions**: Identifying the intent and content of visual elements that standard OCR often misses.
  - **Precise BBox Coordinates**: Mapping visual elements back to their exact location in the original PDF.
- **Sidecar Metadata Mapping**: These visual insights are "stitched" into the metadata of surrounding text chunks, enabling the unique **Sidecar Architecture** for visual grounding.

Each visual element is aligned with its corresponding text chunks to maintain grounding.

---

### 2️⃣ Sidecar Metadata Architecture

A distinctive feature of this pack.

Every document node carries structured metadata such as:

- `file_name`
- `page_label`
- `bbox`
- image references
- visual flags (`has_images`)

This enables downstream applications to:

- Highlight exact PDF regions
- Jump directly to referenced diagrams
- Display visual citations alongside answers

---

### 3️⃣ Dual-Store Hybrid Reasoning

| Component  | Role                                             |
| ---------- | ------------------------------------------------ |
| **Qdrant** | Dense semantic vector retrieval                  |
| **Neo4j**  | Knowledge graph storage and structured reasoning |

This hybrid approach allows:

- Semantic similarity search
- Entity-level reasoning
- Structured dependency traversal
- More robust multi-hop retrieval

---

### 4️⃣ Agentic Retrieval Workflow

Instead of naïve retrieve-and-answer, this pack implements:

```
Retrieve → Grade → Rewrite → (Optional) Web Search Fallback
```

Capabilities include:

- Retrieval quality grading
- Automatic query rewriting
- Optional Tavily-based web fallback
- Improved grounding and answer reliability

---

## 📦 Installation

Install via pip:

```bash
pip install llama-index-packs-multimodal-agentic-rag
```

Or download using `llamaindex-cli`:

```bash
llamaindex-cli download-llamapack MultimodalAgenticRAGPack --download-dir ./multimodal_rag_pack
```

---

## 🛠 Usage

### Initialize the Pack

```python
from llama_index.packs.multimodal_agentic_rag import MultimodalAgenticRAGPack

pack = MultimodalAgenticRAGPack(
    dashscope_api_key="YOUR_DASHSCOPE_KEY",
    qdrant_url="http://localhost:6333",
    neo4j_url="bolt://localhost:7687",
    neo4j_password="your_password",
    tavily_api_key="YOUR_TAVILY_KEY",  # Optional
    data_dir="./parsing_artifacts",
    force_recreate=True,  # WARNING: clears existing DB collections
)
```

---

## 📥 Ingestion Phase

Handles:

- PDF parsing
- Vision-language analysis
- Sidecar metadata generation
- Vector + Graph database upsertion

```python
await pack.run_ingestion("path/to/your/document.pdf")
```

---

## 🔎 Query Phase

Executes the full agentic workflow and returns grounded responses enriched with visual metadata.

```python
query = "Explain the self-attention mechanism in the Transformer architecture."
response = await pack.run(query)

print("AI Answer:", response["final_response"])
```

The response contains:

- Final grounded answer
- Retrieved nodes
- Bounding box metadata
- Source file references

---

## 🧩 Sidecar Metadata Example

Each retrieved node includes metadata like:

```json
{
  "file_name": "transformer.pdf",
  "page_label": "3",
  "bbox": "[[159.3, 140.4, 504.8, 373.9]]",
  "has_images": true
}
```

This enables:

- Precise diagram highlighting
- Visual citation rendering
- Interactive PDF navigation

---

## 🧱 Dependencies

Core components:

- `llama-index`
- `qdrant-client`
- `neo4j`
- `dashscope` (Qwen-VL & Qwen-Plus)
- `pdfplumber`
- `PyMuPDF`

Optional:

- `tavily-python` (web search fallback)

---

## 🏗 Architecture Overview

```
PDF
 ↓
Vision-Language Model (Dense Caption + BBox Extraction)
 ↓
Sidecar Metadata Injection
 ↓
Qdrant (Vector Store) + Neo4j (Graph Store)
 ↓
Agentic Retrieval Workflow
 ↓
Grounded Answer + Visual Highlight Metadata
```

---

## 🎯 Recommended Use Cases

This pack is ideal if:

- Your documents contain critical diagrams or formulas
- You require UI-level visual grounding
- You need hybrid Vector + Graph reasoning
- You want an agentic fallback workflow for robust answers

---

## 📚 Tutorial

For a complete walkthrough, see the accompanying Jupyter Notebook:

`Multimodal_Agentic_RAG_Tutorial.ipynb`

It demonstrates:

- End-to-end ingestion
- Query execution
- Metadata inspection
- Visual grounding usage

---

## License

MIT License
