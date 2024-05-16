from typing import List, Tuple


def default_parse_triplets_fn(
    response: str, max_length: int = 128
) -> List[Tuple[str, str, str]]:
    knowledge_strs = response.strip().split("\n")
    results = []
    for text in knowledge_strs:
        if "(" not in text or ")" not in text or text.index(")") < text.index("("):
            # skip empty lines and non-triplets
            continue
        triplet_part = text[text.index("(") + 1 : text.index(")")]
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
