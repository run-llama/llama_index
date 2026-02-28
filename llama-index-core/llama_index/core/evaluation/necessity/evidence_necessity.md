# Evidence Necessity Evaluation (ENE)

A deterministic, counterfactual evaluation framework for diagnosing **which retrieved evidence is actually necessary** to support factual claims in RAG and LLM-generated responses.

> ENE does not ask *“Does this answer look supported?”*  
> It asks the stronger diagnostic question:  
> **“Which evidence actually matters?”**
**32 tests • 1.88s • unit + property + fuzz + concurrency tested**
---

## Motivation

Most RAG evaluation metrics answer the question:

> *“Does the answer appear to be supported by the retrieved context?”*

This is an **observational** question.

Observational faithfulness is insufficient for diagnosing real RAG failures:

- Redundant contexts can mask hallucinations  
- Spurious grounding can appear “supported”  
- Removing critical evidence is not tested  
- Fragility vs robustness is invisible  

What is missing is a **counterfactual diagnostic question**:

> **Which retrieved contexts are actually necessary for supporting each factual claim?**

Evidence Necessity Evaluation (ENE) answers this question directly.

---

## Core Question

For a given generated response and its retrieved contexts:

> **Which pieces of evidence are counterfactually necessary for each factual claim in the response?**

ENE explicitly distinguishes between:

- Evidence that is present  
- Evidence that is necessary  
- Evidence that is redundant  

---

## Definitions

### Claim
A **claim** is a minimal factual statement extracted deterministically from the model response.

ENE uses sentence-level claims to remain conservative and evaluation-safe.

---

### Supported Claim
A claim is **supported** if it is verified as grounded when *all retrieved contexts are present*.

Support verification is delegated to a deterministic, fail-closed oracle.

---

### Necessary Context
A context is **necessary** for a claim iff:

1. The claim is supported when all contexts are present, **and**  
2. Removing that context causes the claim to become unsupported  

This definition encodes a counterfactual intervention.

---

### Fragile vs Robust Claims

- **Fragile claim**  
  Depends on exactly one necessary context  
  → single-point failure  

- **Robust claim**  
  Depends on multiple necessary contexts  
  → evidence redundancy  

---

## Method

ENE follows a deterministic, bounded pipeline.

### 1. Claim Extraction
The response is split into atomic claims using deterministic sentence segmentation.  
No LLMs are used.

---

### 2. Baseline Support Check
Each claim is verified against the full context set.

If a claim is unsupported at baseline:
- It is marked unsupported  
- No counterfactual analysis is performed  

Fail-closed semantics apply.

---

### 3. Counterfactual Necessity Analysis
For each supported claim:
- Each context is removed **one at a time**  
- The claim is re-verified  
- If removal causes failure, the context is marked necessary  

This yields a minimal necessity set per claim.

---

### 4. Evidence Necessity Graph (ENG)
Results are assembled into a **structural artifact**:

- Claims → nodes  
- Context indices → nodes  
- Edges encode necessity relations  

No scalar scoring is produced.

---

## Design Principles

ENE is intentionally conservative, minimal, and extensible.  
Its purpose is not to be clever; its purpose is to be **correct**.

### Hard Guarantees
- **Deterministic** — Temperature-0 execution, cache-safe, zero sampling variance  
- **Fail-Closed** — False negatives preferred over false positives  
- **Bounded** — Exactly `1 + N` oracle calls per claim  
- **Evaluation-Safe** — Pure analysis artifact, no effect on training or inference  

---

## Engineering Guarantees

ENE is engineered as evaluation infrastructure with explicit, test-backed guarantees:

- **Deterministic execution**  
  Temperature-0 oracle calls, cache-safe behavior, no stochastic branching.

- **Bounded oracle complexity**  
  Exactly **O(1 + N)** oracle calls per claim.

- **Bounded concurrency**  
  All oracle calls executed under a strict semaphore-enforced concurrency cap.

- **Schema-versioned artifacts**  
  Evidence Necessity Graphs (ENG) are versioned and round-trip safe.

- **Defensive parsing & validation**  
  Malformed inputs, oracle ambiguity, and deserialization errors fail fast and fail closed.

All guarantees are enforced via:
- Unit tests  
- Property-based tests (Hypothesis)  
- Fuzz tests  
- Concurrency invariant tests  

---

## Non-Goals

ENE intentionally does **not**:

- Assign confidence scores or rank claims  
- Score answer quality  
- Re-evaluate model reasoning  
- Replace probabilistic faithfulness metrics  

> ENE exposes **structure**, not judgment.

---

## Comparison with Existing Tools

| Tool | Approach | Counterfactual | Deterministic | Structural |
|------|----------|----------------|---------------|------------|
| RAGAS | Observational | ❌ | ❌ | ❌ |
| TruLens | Observational | ❌ | ❌ | ❌ |
| LangSmith | Heuristic | ❌ | ❌ | ❌ |
| **ENE (Ours)** | **Counterfactual** | ✅ | ✅ | ✅ |

> ENE complements existing tools — it does not replace them.

---

## Technical Specifications

### Computational Complexity

For a response with:
- `C` claims  
- `N` contexts  

Worst-case oracle calls:

`O(C x (1 + N))`


Execution is parallelizable under a strict concurrency cap.

---

### Explicit Failure Modes

ENE makes failure **visible rather than cosmetic**:

- **Unsupported claims** are surfaced explicitly  
- **Redundant contexts** are identified  
- **Fragile claims** are flagged  
- **Oracle uncertainty** propagates safely  

This makes evaluation **diagnostic, not decorative**.

---

## Example (Conceptual)

**Response**  
> "The Eiffel Tower is in Paris."

**Contexts**  
1. `[0]` "The Eiffel Tower is located in Paris."  
2. `[1]` "The Eiffel Tower is a tourist attraction."

**ENE Result**
- Claim supported  
- Context `[0]` necessary  
- Context `[1]` redundant  
- Claim classified as **Fragile**

---

## Use Cases

- **Debugging Retrieval Pipelines**  
  Identify which chunks truly support which claims.

- **Diagnosing Hallucinations**  
  Detect when the model relies on internal knowledge instead of context.

- **Measuring Redundancy**  
  Reduce token waste by identifying unused contexts.

- **Building Trust**  
  Provide human-legible evidence structures for evaluation.

---

## Summary

Evidence Necessity Evaluation introduces **counterfactual, deterministic evaluation** into RAG systems.

It answers a question existing tools cannot:

> **Which evidence actually matters?**

This makes ENE especially suitable for:
- Retrieval debugging  
- Hallucination diagnosis  
- Evidence redundancy analysis  
- Trustworthy evaluation tooling  
