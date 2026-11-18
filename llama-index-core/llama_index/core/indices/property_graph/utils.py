from typing import List, Tuple


def _extract_triplet_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                candidates.append(text[start:i])
                start = -1
    if not candidates and start != -1:
        close_idx = text.rfind(")")
        if close_idx >= start:
            candidates.append(text[start:close_idx])
    return candidates


def default_parse_triplets_fn(
    response: str, max_length: int = 128
) -> List[Tuple[str, str, str]]:
    knowledge_strs = response.strip().split("\n")
    results = []
    for text in knowledge_strs:
        for triplet_part in _extract_triplet_candidates(text):
            tokens = triplet_part.split(",")
            if len(tokens) != 3:
                continue

            if any(len(s.encode("utf-8")) > max_length for s in tokens):
                # We count byte-length instead of len() for UTF-8 chars,
                # will skip if any of the tokens are too long.
                # This is normally due to a poorly formatted triplet
                # extraction, in more serious KG building cases
                # we'll need NLP models to better extract triplets.
                continue

            subj, pred, obj = map(str.strip, tokens)
            if not subj or not pred or not obj:
                # skip partial triplets
                continue

            # Strip double quotes and Capitalize triplets for disambiguation
            subj, pred, obj = (
                entity.strip('"').capitalize() for entity in [subj, pred, obj]
            )

            results.append((subj, pred, obj))
    return results
