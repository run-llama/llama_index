from llama_index.core.node_parser.text import TokenTextSplitter
from llama_index.core.schema import Document


def test_token_splitter_accepts_none_backup_separators() -> None:
    """Fix #23790: backup_separators=None must not raise ValidationError.

    The constructor annotates backup_separators as Optional[List[str]], but the
    Pydantic field was declared as List, so None was rejected during init before
    the splitter could be created.
    """
    splitter = TokenTextSplitter(tokenizer=list, backup_separators=None)
    nodes = splitter.get_nodes_from_documents(
        [Document(text="alpha beta gamma delta epsilon")]
    )
    assert len(nodes) >= 1
    # None is accepted at the field layer (and treated as no extra separators downstream).
    assert splitter.backup_separators is None or splitter.backup_separators == []


def test_token_splitter_default_backup_separators_preserved() -> None:
    """The default backup_separators (["\\n"]) is preserved when not overridden."""
    splitter = TokenTextSplitter(tokenizer=list)
    assert splitter.backup_separators == ["\n"]
    nodes = splitter.get_nodes_from_documents(
        [Document(text="line one\nline two\nline three")]
    )
    assert len(nodes) >= 1
