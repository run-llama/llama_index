from llama_index.core.node_parser.text import TokenTextSplitter
from llama_index.core.schema import Document


def test_token_splitter_accepts_none_backup_separators() -> None:
    """
    Fix #23790: backup_separators=None must not raise ValidationError.

    The constructor annotates backup_separators as Optional[List[str]], but the
    Pydantic field was declared as List, so None was rejected during init before
    the splitter could be created.
    """
    splitter = TokenTextSplitter(tokenizer=list, backup_separators=None)
    nodes = splitter.get_nodes_from_documents(
        [Document(text="alpha beta gamma delta epsilon")]
    )
    assert len(nodes) >= 1
    assert splitter.backup_separators == []


def test_token_splitter_default_backup_separators_preserved() -> None:
    """The default newline backup separator is preserved when not overridden."""
    splitter = TokenTextSplitter(tokenizer=list)
    assert splitter.backup_separators == ["\n"]
    nodes = splitter.get_nodes_from_documents(
        [Document(text="line one\nline two\nline three")]
    )
    assert len(nodes) >= 1


def test_none_and_empty_backup_separators_split_consistently() -> None:
    none_splitter = TokenTextSplitter(
        chunk_size=5, chunk_overlap=0, tokenizer=list, backup_separators=None
    )
    empty_splitter = TokenTextSplitter(
        chunk_size=5, chunk_overlap=0, tokenizer=list, backup_separators=[]
    )
    assert none_splitter.split_text("alpha beta") == ["alpha", "beta"]
    assert none_splitter.split_text("alpha beta") == empty_splitter.split_text(
        "alpha beta"
    )


def test_from_defaults_accepts_none_backup_separators() -> None:
    splitter = TokenTextSplitter.from_defaults(backup_separators=None)
    assert splitter.backup_separators == []
    assert splitter.split_text("alpha beta") == ["alpha beta"]
