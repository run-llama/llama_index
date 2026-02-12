import os
import re
import fitz  # PyMuPDF
import pdfplumber
import dashscope
from pathlib import Path
from http import HTTPStatus
from llama_index.core import Document
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
import base64
import hashlib
import tempfile
import io
from PIL import Image
from collections import Counter
import logging
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EduMatrixPDFReader")

"""
SIDECAR SCHEMA (page_heavy_data.json)
-------------------------------------
{
    "file_name_p1": {
        "graph_data": {
            "entities": [{"name": str, "category": str, ...}],
            "relations": [{"source": str, "target": str, "relation": str}]
        },
        "evidence_images": [
            {"path": str, "bbox": list, "type": str, "caption": str}
        ],
        "raw_text_len": int
    },
    ...
}

Why Sidecar?
1. Vector Efficiency: Keeps vector store payloads small for faster retrieval.
2. Evidence Traceability: Allows UI to fetch original image crops via BBox.
3. Graph Portability: Simplifies export to Neo4j or other GraphDBs.
"""

DEFAULT_VLM_PROMPT = (
    "You are an expert Teaching Assistant. Analyze this image for **Knowledge Graph Construction** and **Exam Review**.\n"
    "Your tasks: 1. Filter Noise 2. Extract Structured Knowledge.\n\n"
    "**Step 1: Classification**\n"
    "- 'NOISE': Logos, page numbers, decorative headers/footers, blurry icons, pure background, watermarks, generic clip art, silhouettes, stock photos with no educational value.\n"
    "- 'ARCH': System architecture, block diagrams.\n"
    "- 'FLOW': Algorithms, flowcharts, sequence diagrams.\n"
    "- 'CODE': Code snippets, terminal outputs.\n"
    "- 'FORMULA': Mathematical equations (highly important).\n"
    "- 'CHART/TABLE': Data visualizations.\n"
    "- 'OTHER': Only for meaningful illustrations that directly explain a CS concept (e.g., a photo of a server rack). If it's just a decorative icon, classify as NOISE.\n\n"
    "**Step 2: Output JSON**\n"
    "```json\n"
    "{\n"
    '  "type": "(ARCH | FLOW | CODE | FORMULA | CHART | TABLE | OTHER | NOISE)",\n'
    '  "dense_caption": "Detailed description. For ARCH/FLOW, describe the data flow. For FORMULA, provide LaTeX.",\n'
    '  "keywords": ["key1", "key2"],\n'
    '  "entities": [\n'
    '    {"name": "Concept Name", "canonical_name": "Full Name (no abbr)", "category": "(Component|Algorithm|Metric|Concept)"}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"source": "A", "target": "B", "relation": "(INCLUDES|FLOWS_TO|CALCULATES|DEPENDS_ON)", "description": "context"}\n'
    "  ],\n"
    '  "ocr_text": "Exact text in the image"\n'
    "}\n"
    "```\n"
    "Constraint: If type is NOISE, return empty lists for entities/relations.\n"
    "**Critical Rule**: If the image has a figure number (e.g., 'Figure 3-1', '图 6-2'), "
    "you MUST start your description with: 'This is [Figure X-Y]: ...'. "
    "Make sure the figure number is exactly as it appears."
)


class EduMatrixPDFReader:
    def __init__(
        self,
        pdf_path: str,
        image_output_dir: str = "../data/parsed_images",
        cache_file: str = "../data/vlm_cache.json",
        hash_record_file: str = "../data/processed_hashes.json",
        sidecar_file: str = "../data/page_heavy_data.json",
        embedding_cache_file: str = "../data/embedding_cache.json",
        alias_map_file: str = "../data/global_alias_map.json",
        use_vlm: bool = True,
        vlm_model_name: str = "qwen-vl-max",
        vlm_system_prompt: str = DEFAULT_VLM_PROMPT,
        max_concurrency: int = 5,
        min_image_bytes: int = 3072,
        max_edge_size: int = 1600,
        min_edge_size: int = 100,
        jpeg_quality: int = 85,
    ):
        self.pdf_path = Path(pdf_path)
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        # Persistence Paths
        self.cache_file = Path(cache_file)
        self.hash_record_file = Path(hash_record_file)
        self.sidecar_file = Path(sidecar_file)
        self.embedding_cache_file = Path(embedding_cache_file)
        self.alias_map_file = Path(alias_map_file)

        # Configuration
        self.use_vlm = use_vlm
        self.model_name = vlm_model_name
        self.system_prompt = vlm_system_prompt
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.min_image_bytes = min_image_bytes
        self.max_edge_size = max_edge_size
        self.min_edge_size = min_edge_size
        self.jpeg_quality = jpeg_quality

        # [Check] Validate API Key immediately. Degrade gracefully if missing.
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if (self.use_vlm) and not self.api_key:
            logger.warning(
                "⚠️ DASHSCOPE_API_KEY not found. VLM and Embeddings will be DISABLED."
            )
            self.use_vlm = False

        # Load Persistent State
        self.cache_data = self._load_json(self.cache_file)
        self.global_processed_imgs = set(
            self._load_json(self.hash_record_file).get("hashes", [])
        )
        self.page_heavy_data = self._load_json(self.sidecar_file)
        self.embedding_map = self._load_json(self.embedding_cache_file)

        # Load learned aliases (e.g., "CPU" -> "Central Processing Unit")
        self.global_alias_map: Dict[str, str] = self._load_json(self.alias_map_file)

        self.VALID_TYPES = {
            "ARCH",
            "FLOW",
            "CODE",
            "FORMULA",
            "CHART",
            "TABLE",
            "OTHER",
        }

    def _load_json(self, path: Path) -> Dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                return {}
        return {}

    def _save_data(self):
        """Atomic write for all state files to prevent corruption during concurrency."""
        self._atomic_write(self.cache_file, self.cache_data)
        self._atomic_write(
            self.hash_record_file, {"hashes": list(self.global_processed_imgs)}
        )
        self._atomic_write(self.sidecar_file, self.page_heavy_data)
        self._atomic_write(self.embedding_cache_file, self.embedding_map)
        self._atomic_write(self.alias_map_file, self.global_alias_map)

    def _atomic_write(self, path: Path, data: Any):
        """
        Performs a thread-safe and crash-safe atomic write to a JSON file.
        Prevents data corruption by writing to temp first.
        """
        tmp_path = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, text=True)
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.error(f"⚠️ Data save failed {path.name}: {e}")

    # --- Image Processing Helpers ---
    def _compute_semantic_hash(self, img_bytes: bytes) -> str:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # Convert to Grayscale & Resize to ensure hash stability
                img = img.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
                return hashlib.sha256(img.tobytes()).hexdigest()
        except Exception:
            return hashlib.md5(img_bytes).hexdigest()

    def _resize_and_compress_image(self, img_bytes: bytes) -> bytes:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                w, h = img.size
                if max(w, h) > self.max_edge_size:
                    scale = self.max_edge_size / max(w, h)
                    img = img.convert("RGB").resize(
                        (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                    )
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=self.jpeg_quality)
                    return out.getvalue()
                return img_bytes
        except Exception:
            return img_bytes

    # --- Text Processing Helpers ---
    def _table_to_markdown(self, table):
        if not table or len(table) < 1:
            return ""
        cleaned = [
            [str(cell).replace("\n", " ") if cell is not None else "" for cell in row]
            for row in table
        ]
        if not cleaned:
            return ""
        header = "| " + " | ".join(cleaned[0]) + " |"
        sep = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
        if len(cleaned) > 1:
            rows = ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
            return f"\n\n{header}\n{sep}\n" + "\n".join(rows) + "\n\n"
        return f"\n\n{header}\n{sep}\n\n"

    def _extract_json(self, text: str) -> Dict:
        """Robust JSON extraction from LLM response."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback 1: Regex
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Fallback 2: Brute force braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        return {
            "dense_caption": text,
            "entities": [],
            "relations": [],
            "keywords": [],
            "type": "UNKNOWN",
        }

    def _extract_text_keywords(self, text: str, top_k: int = 15) -> List[str]:
        """
        Extracts keywords using title analysis and technical term frequency.
        """
        keywords_pool = []
        if not text:
            return []

        # 1. Title Priority
        title = self._extract_section_title(text)
        if title:
            clean = re.sub(
                r"^[\d\.]+\s*|第[一二三四五六七八九十\d]+[章节]\s*", "", title
            ).strip()
            if clean:
                # Weight title heavily
                keywords_pool.extend([clean] * 5)

        # 2. Technical Term Extraction (Capitalized Words)
        english_terms = re.findall(r"\b[A-Z][A-Za-z0-9_]{1,}\b", text)
        if english_terms:
            STOP_WORDS = {
                "THE",
                "AND",
                "FOR",
                "FIG",
                "TABLE",
                "PAGE",
                "SECTION",
                "CHAPTER",
                "IMAGE",
                "FIGURE",
            }
            filtered = [
                t for t in english_terms if t.upper() not in STOP_WORDS and len(t) > 1
            ]
            keywords_pool.extend(filtered)

        # 3. Frequency Ranking
        counts = Counter(keywords_pool)

        return [word for word, count in counts.most_common(top_k)]

    def _sanitize_canonical_name(self, name: str, canonical: str) -> str:
        """Sanitize names to prevent LLM hallucination length issues."""
        if not canonical:
            return name
        clean_can = canonical.strip()
        clean_name = name.strip()

        # Reject if canonical is too long or looks like a sentence
        if len(clean_can) > 40 or len(clean_can) > len(clean_name) * 4:
            return clean_name
        if " " in clean_can:
            words = clean_can.split()
            if len(set(words)) < len(words) * 0.5:
                return clean_name
        return clean_can

    def _extract_section_title(self, text: str) -> Optional[str]:
        if not text:
            return None
        for line in text.split("\n")[:5]:
            line = line.strip()
            if len(line) > 60:
                continue
            if re.match(r"^\d+(\.\d+)*\s+.+", line) or re.match(
                r"^(Chapter|Section|Part|第[一二三四五六七八九十\d]+[章节])",
                line,
                re.IGNORECASE,
            ):
                return line
        return None

    # --- Embedding & Vector Logic ---
    async def _get_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            return []

        if text in self.embedding_map:
            return self.embedding_map[text]

        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: dashscope.TextEmbedding.call(
                    model=dashscope.TextEmbedding.Models.text_embedding_v2,
                    input=text,
                    api_key=self.api_key or "",
                ),
            )
            if resp.status_code == HTTPStatus.OK:
                emb = resp.output["embeddings"][0]["embedding"]
                self.embedding_map[text] = emb
                return emb
            else:
                logger.warning(f"Embedding API failed for '{text}': {resp.message}")
        except Exception as e:
            logger.error(f"Embedding exception for '{text}': {e}")
        return []

    def _cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        # Dimension mismatch check
        if len(vec1) != len(vec2):
            if not hasattr(self, "_dim_error_logged"):
                logger.error(f"Dimension mismatch: {len(vec1)} vs {len(vec2)}.")
                self._dim_error_logged = True
            return 0.0

        v1, v2 = np.array(vec1), np.array(vec2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        return np.dot(v1, v2) / norm_product

    async def _dedup_graph_data(
        self, entities: List[Dict], relations: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Synchronizes local page entities with global knowledge base using
        Hybrid Alignment (String Match + Vector Similarity).
        """
        unique_entities = []
        seen_entities = set()

        # --- 1. Entity Alignment ---
        for e in entities:
            name = e.get("name", "").strip()
            raw_canonical = e.get("canonical_name", "").strip()
            category = e.get("category", "Concept").strip()

            if not name or len(name) < 2:
                continue

            current_name = self._sanitize_canonical_name(name, raw_canonical)
            name_key = current_name.upper()

            # A. Fast Path: String Match
            if name_key in self.global_alias_map:
                final_name = self.global_alias_map[name_key]
            else:
                # B. Slow Path: Vector Alignment
                final_name = current_name

                if (
                    self.api_key
                    and len(self.embedding_map) > 0
                    and len(self.global_alias_map) < 8000
                ):
                    current_vec = await self._get_embedding(current_name)

                    if current_vec:
                        best_match: str = ""
                        highest_sim = 0.0

                        for existing_name, existing_vec in self.embedding_map.items():
                            if existing_name == current_name:
                                continue
                            if len(existing_name.split()) > 6:
                                continue

                            sim = self._cosine_similarity(current_vec, existing_vec)
                            if sim > highest_sim:
                                highest_sim = sim
                                best_match = existing_name
                                if highest_sim > 0.99:
                                    break

                        # Threshold 0.92
                        if highest_sim > 0.92 and best_match:
                            logger.info(
                                f"🧬 Semantic Match: '{current_name}' -> '{best_match}' (sim: {highest_sim:.4f})"
                            )
                            final_name = best_match
                            self.global_alias_map[name_key] = final_name
                        else:
                            self.global_alias_map[name_key] = current_name
                    else:
                        self.global_alias_map[name_key] = current_name
                else:
                    self.global_alias_map[name_key] = current_name

            e["name"] = final_name
            if name != final_name:
                e["original_name"] = name

            key = (final_name.upper(), category.upper())
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(e)

        # --- 2. Relation Alignment ---
        unique_relations = []
        seen_relations = set()

        for r in relations:
            src = str(r.get("source", "") or "").strip()
            tgt = str(r.get("target", "") or "").strip()
            rel = str(r.get("relation", "") or "").strip()

            if not src or not tgt:
                continue
            if src.upper() == tgt.upper():
                continue

            final_src = self.global_alias_map.get(src.upper(), src)
            final_tgt = self.global_alias_map.get(tgt.upper(), tgt)

            r["source"] = final_src
            r["target"] = final_tgt
            if "weight" not in r:
                r["weight"] = 1.0

            key = (final_src.upper(), final_tgt.upper(), rel.upper())
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)

        return unique_entities, unique_relations

    # --- VLM Interaction ---
    async def _describe_image_with_retry(self, img_bytes: bytes, img_hash: str):
        if img_hash in self.cache_data:
            cached = self.cache_data[img_hash]
            if cached.get("type") in ["FAILED", "NOISE"]:
                return None
            if (
                cached.get("type") in self.VALID_TYPES
                or cached.get("type") == "UNKNOWN"
            ):
                return cached

        if not self.use_vlm:
            return None

        processed_bytes = self._resize_and_compress_image(img_bytes)
        b64_data = base64.b64encode(processed_bytes).decode("utf-8")
        img_data_uri = f"data:image/jpeg;base64,{b64_data}"

        messages = [
            {
                "role": "user",
                "content": [{"image": img_data_uri}, {"text": self.system_prompt}],
            }
        ]

        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: dashscope.MultiModalConversation.call(
                            model=self.model_name,
                            messages=messages,
                            api_key=self.api_key or "",
                        ),
                    )

                    from typing import Any, cast

                    valid_resp = cast(Any, response)

                    if valid_resp.status_code == HTTPStatus.OK:
                        content = valid_resp.output.choices[0].message.content
                        if isinstance(content, list):
                            text = "\n".join(
                                c.get("text", "")
                                for c in content
                                if isinstance(c, dict)
                            )
                        else:
                            text = str(content)

                        if text:
                            json_data = self._extract_json(text)
                            result_type = json_data.get("type", "UNKNOWN")

                            ban_words = {
                                "silhouette",
                                "logo",
                                "icon",
                                "watermark",
                                "copyright",
                                "advertisement",
                                "stock photo",
                                "decorative",
                            }
                            caption_lower = json_data.get("dense_caption", "").lower()
                            keywords_lower = {
                                k.lower() for k in json_data.get("keywords", [])
                            }

                            # Heuristic Filter
                            if any(w in caption_lower for w in ban_words) or any(
                                w in keywords_lower for w in ban_words
                            ):
                                logger.info(
                                    f"🗑️ Heuristic Filter: Forced NOISE for img={img_hash[:8]}"
                                )
                                result_type = "NOISE"
                                json_data["type"] = "NOISE"
                                json_data["entities"] = []
                                json_data["relations"] = []

                            if (
                                result_type in self.VALID_TYPES
                                or result_type == "UNKNOWN"
                            ):
                                self.cache_data[img_hash] = json_data
                                return json_data
                            elif result_type == "NOISE":
                                self.cache_data[img_hash] = json_data
                                return None
                    else:
                        logger.warning(
                            f"API Error: {valid_resp.code} - {valid_resp.message}"
                        )

                except Exception as e:
                    logger.warning(
                        f"[VLM ERROR] img={img_hash[:8]} attempt={attempt + 1} err={e}"
                    )
                    if attempt < 2:
                        await asyncio.sleep(2**attempt)

            self.cache_data[img_hash] = {"type": "FAILED"}
            return None

    # --- Main Pipeline ---
    async def parse(self) -> List[Document]:
        """
        Main parsing pipeline: PDF -> Text/Tables/Images -> Sidecar Data -> LlamaIndex Docs.
        """
        logger.info(f"🐢 [EduMatrix Parser] Starting: {self.pdf_path.name}")
        pages_content: Dict[int, Dict[str, Any]] = {}
        all_image_tasks = []

        try:
            doc_fitz = fitz.open(self.pdf_path)
            total_pages = len(doc_fitz)
        except Exception as e:
            logger.error(f"Failed to open PDF {self.pdf_path}: {e}")
            return []

        for i in range(total_pages):
            pages_content[i] = {
                "raw_text": "",
                "text_parts": [],
                "graph_entities": [],
                "graph_relations": [],
                "page_keywords": [],
                "image_meta": [],
                "evidence_images": [],
            }

        logger.info("    📖 Extracting text and tables...")

        def extract_text():
            try:
                with pdfplumber.open(self.pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        if i >= total_pages:
                            break
                        text = page.extract_text()
                        if text:
                            pages_content[i]["raw_text"] = text
                            pages_content[i]["text_parts"].append(text)

                        tables = page.extract_tables()
                        for idx, tbl in enumerate(tables):
                            md = self._table_to_markdown(tbl)
                            if md:
                                pages_content[i]["text_parts"].append(
                                    f"\n=== Table {idx + 1} ===\n{md}"
                                )
            except Exception as e:
                logger.error(f"PDFPlumber error: {e}")

        await asyncio.get_event_loop().run_in_executor(None, extract_text)

        logger.info("    🖼️ Extracting images...")
        for i in range(total_pages):
            page_fitz = doc_fitz[i]
            image_list = page_fitz.get_images(full=True)

            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    rects = page_fitz.get_image_rects(xref)
                    bbox = [float(x) for x in rects[0]] if rects else []
                    base_image = doc_fitz.extract_image(xref)
                    img_bytes = base_image["image"]

                    w, h = base_image.get("width", 0), base_image.get("height", 0)
                    if (
                        len(img_bytes) < self.min_image_bytes
                        or w < self.min_edge_size
                        or h < self.min_edge_size
                    ):
                        continue

                    aspect_ratio = w / h
                    if aspect_ratio > 10 or aspect_ratio < 0.1:
                        continue

                    try:
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            gray_img = pil_img.convert("L")
                            extrema = gray_img.getextrema()

                            # [Fix] Explicit type check for tuple
                            if isinstance(extrema, tuple) and len(extrema) == 2:
                                mn, mx = extrema
                                # Double check numbers to avoid Pylance issues
                                if isinstance(mn, (int, float)) and isinstance(
                                    mx, (int, float)
                                ):
                                    if mx - mn < 10:
                                        continue
                    except Exception:
                        pass

                    img_hash = self._compute_semantic_hash(img_bytes)
                    img_ext = base_image["ext"]
                    img_path = (
                        self.image_output_dir
                        / f"p{i + 1}_{img_idx}_{img_hash[:8]}.{img_ext}"
                    )

                    if not img_path.exists():
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)

                    task = self._describe_image_with_retry(img_bytes, img_hash)
                    all_image_tasks.append(task)

                    pages_content[i]["image_meta"].append(
                        {
                            "task_idx": len(all_image_tasks) - 1,
                            "path": str(img_path),
                            "bbox": bbox,
                            "hash": img_hash,
                            "local_idx": img_idx,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Image extract error p{i}: {e}")
        doc_fitz.close()

        image_results: List[Any] = [None] * len(all_image_tasks)
        if all_image_tasks:
            logger.info(f"    🚀 Processing {len(all_image_tasks)} images...")
            results = await asyncio.gather(*all_image_tasks, return_exceptions=True)

            for j, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error(f"Task failed: {res}")
                    image_results[j] = None
                else:
                    image_results[j] = res

        final_documents = []
        current_section_title = "General"

        for i in range(total_pages):
            page_data = pages_content[i]
            title = self._extract_section_title(page_data["raw_text"])
            if title:
                current_section_title = title

            text_kws = self._extract_text_keywords(page_data["raw_text"])
            page_data["page_keywords"].extend(text_kws)

            for meta in page_data["image_meta"]:
                res = image_results[meta["task_idx"]]
                if isinstance(res, Exception) or not res:
                    continue

                if res.get("type") in ["NOISE", "NONE", "UNKNOWN", "FAILED"]:
                    continue

                caption_block = (
                    f"\n>>> [IMAGE: {res.get('type')}]\n"
                    f"Caption: {res.get('dense_caption')}\n"
                    f"OCR: {res.get('ocr_text')}\n"
                    f"<<<\n"
                )
                page_data["text_parts"].append(caption_block)

                # Limit image keywords
                if res.get("keywords"):
                    page_data["page_keywords"].extend(res.get("keywords")[:3])

                if meta["hash"] not in self.global_processed_imgs:
                    self.global_processed_imgs.add(meta["hash"])

                    if res.get("entities"):
                        for e in res.get("entities"):
                            e.update({"source": "image", "page": i + 1})
                        page_data["graph_entities"].extend(res.get("entities"))

                    if res.get("relations"):
                        for r in res.get("relations"):
                            r.update({"provenance": "image", "page": i + 1})
                        page_data["graph_relations"].extend(res.get("relations"))

                page_data["evidence_images"].append(
                    {
                        "path": meta["path"],
                        "bbox": meta["bbox"],
                        "type": res.get("type"),
                        "caption": res.get("dense_caption"),
                    }
                )

            kw_counter = Counter(page_data["page_keywords"])
            top_kws = [k for k, v in kw_counter.most_common(15)]
            keywords_str = ", ".join(top_kws)

            full_text = (
                f"[SECTION] {current_section_title}\n"
                f"[KEYWORDS] {keywords_str}\n"
                f"=== Page {i + 1} ===\n" + "\n".join(page_data["text_parts"])
            )

            # Async Deduplication
            u_ents, u_rels = await self._dedup_graph_data(
                page_data["graph_entities"], page_data["graph_relations"]
            )

            page_id = f"{self.pdf_path.name}_p{i + 1}"

            self.page_heavy_data[page_id] = {
                "graph_data": {"entities": u_ents, "relations": u_rels},
                "evidence_images": page_data["evidence_images"],
                "raw_text_len": len(page_data["raw_text"]),
            }

            final_documents.append(
                Document(
                    text=full_text,
                    metadata={
                        "file_name": self.pdf_path.name,
                        "page_label": str(i + 1),
                        "section_title": current_section_title,
                        "content_type": "page_compound",
                        "page_id_key": page_id,
                        "has_images": len(page_data["evidence_images"]) > 0,
                    },
                )
            )

        self._save_data()
        logger.info(
            f"✅ Parsing complete! Generated {len(final_documents)} slim document chunks."
        )
        return final_documents
