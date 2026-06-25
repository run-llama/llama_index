# Educational Brief: Stripping Nova `<thinking>` Tags in BedrockConverse

## Step 1 — Real-World Analogy

Imagine a waiter delivering your meal. In the kitchen, the chef scribbles private prep notes on the order ticket: *"Let me double-check the sauce temperature before plating."* Those notes help the kitchen staff, but they are not meant for the customer.

When Nova models reason through a problem, they sometimes embed similar private notes directly in the response text, wrapped in `<thinking>...</thinking>` tags. Without filtering, BedrockConverse would read those kitchen notes aloud — serving internal reasoning as part of the final answer.

The fix is a simple front-of-house rule: before presenting any response to the caller, remove everything between `<thinking>` and `</thinking>`, including the tags themselves.

---

## Step 2 — LaTeX Formalization

Let \(T\) be the raw text returned by the model. Define the tag delimiters:

\[
\text{open} = \texttt{<thinking>}, \quad \text{close} = \texttt{</thinking>}
\]

Let \(R\) be the regular expression that matches a (case-insensitive) thinking block:

\[
R = \texttt{<thinking>.*?</thinking>} \quad \text{(with DOTALL enabled)}
\]

The cleaning function \(f : \Sigma^* \to \Sigma^*\) is defined as:

\[
f(T) = \text{remove\_orphans}\bigl(\text{remove\_unclosed}(R(T))\bigr)
\]

where:

\[
\text{remove\_unclosed}(S) =
\begin{cases}
S[{:}\text{start}(S)) & \text{if } \exists\, i : S[i{:}] \text{ matches } \texttt{<thinking>.*$} \\
S & \text{otherwise}
\end{cases}
\]

\[
\text{remove\_orphans}(S) = \text{replace}(\text{replace}(S, \texttt{</thinking>}, \varepsilon), \texttt{<thinking>}, \varepsilon)
\]

For streaming, let chunks \(c_1, c_2, \ldots, c_n\) arrive incrementally. Accumulate raw text:

\[
T_k = T_{k-1} \mathbin{\|} c_k, \quad T_0 = \varepsilon
\]

Emit cleaned deltas by differencing successive clean states:

\[
\Delta_k = f(T_k)\bigl[\,|f(T_{k-1})|\,:\,\bigr]
\]

This preserves correct incremental output while tags may span multiple chunks.

---

## Step 3 — Functional Python Demonstration

```python
import re

_THINKING_TAG_PATTERN = re.compile(
    r"<thinking>.*?</thinking>",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking_tags(text: str) -> str:
    """Remove embedded Nova thinking blocks from model text."""
    if not text:
        return text

    cleaned = _THINKING_TAG_PATTERN.sub("", text)

    unclosed = re.search(r"<thinking>.*$", cleaned, re.DOTALL | re.IGNORECASE)
    if unclosed:
        cleaned = cleaned[: unclosed.start()]

    cleaned = re.sub(r"</thinking>", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"<thinking>", "", cleaned, flags=re.IGNORECASE)


def stream_clean_deltas(chunks: list[str]) -> list[str]:
    """Compute cleaned streaming deltas from raw text chunks."""
    prev_clean = ""
    deltas: list[str] = []

    raw = ""
    for chunk in chunks:
        raw += chunk
        clean = strip_thinking_tags(raw)
        deltas.append(clean[len(prev_clean) :])
        prev_clean = clean

    return deltas


# --- Example usage ---
raw_response = (
    "<thinking>Let me reason about this step by step.</thinking>"
    "The capital of France is Paris."
)
assert strip_thinking_tags(raw_response) == "The capital of France is Paris."

stream_chunks = [
    "<thinking>Let me reason",
    " about this step by step.</thinking>",
    "The capital of France is Paris.",
]
assert "".join(stream_clean_deltas(stream_chunks)) == "The capital of France is Paris."
assert "<thinking>" not in "".join(stream_clean_deltas(stream_chunks))
```

This mirrors the production logic in `llama_index/llms/bedrock_converse/base.py` and resolves [issue #20489](https://github.com/run-llama/llama_index/issues/20489).
