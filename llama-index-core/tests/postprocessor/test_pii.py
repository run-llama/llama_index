"""Tests for NERPIINodePostprocessor."""

import re

from llama_index.core.postprocessor.pii import NERPIINodePostprocessor


def _fake_ner(pattern_to_group):
    def ner(text: str):
        out = []
        for pat, grp in pattern_to_group.items():
            for match in re.finditer(pat, text):
                out.append(
                    {
                        "entity_group": grp,
                        "word": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        return out

    return ner


def test_ner_pii_mask_does_not_corrupt_substrings_of_other_words() -> None:
    processor = NERPIINodePostprocessor()
    text = "Al met Alice in Paris."
    masked, mapping = processor.mask_pii(
        _fake_ner({r"\bAl\b": "PER", r"\bAlice\b": "PER", r"Paris": "LOC"}),
        text,
    )
    assert masked == "[PER_0] met [PER_7] in [LOC_16]."
    assert mapping == {"[PER_0]": "Al", "[PER_7]": "Alice", "[LOC_16]": "Paris"}


def test_ner_pii_every_mapping_tag_appears_once_and_roundtrips() -> None:
    processor = NERPIINodePostprocessor()
    text = "Alice lives in Paris, Alice likes Paris."
    masked, mapping = processor.mask_pii(
        _fake_ner({r"Alice": "PER", r"Paris": "LOC"}),
        text,
    )
    assert len(mapping) == 4
    for tag in mapping:
        assert masked.count(tag) == 1
    restored = masked
    for tag, word in mapping.items():
        restored = restored.replace(tag, word)
    assert restored == text
    assert "Alice" not in masked
    assert "Paris" not in masked
