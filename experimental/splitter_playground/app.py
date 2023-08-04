import os
import tempfile
from typing import List

import streamlit as st
import tiktoken
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from llama_index import SimpleDirectoryReader
from llama_index.schema import Document
from llama_index.text_splitter import CodeSplitter, SentenceSplitter, TokenTextSplitter

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."

text = st.sidebar.text_area("Enter text", value=DEFAULT_TEXT)
uploaded_files = st.sidebar.file_uploader("Upload file", accept_multiple_files=True)


@st.cache_resource(ttl="1h")
def load_document(uploaded_files) -> List[Document]:
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

    reader = SimpleDirectoryReader(input_dir=temp_dir.name)
    docs = reader.load_data()
    return docs


if uploaded_files:
    if text != DEFAULT_TEXT:
        st.warning("Text will be ignored when uploading files")
    docs = load_document(uploaded_files)
    text = "\n".join([doc.text for doc in docs])


type = st.sidebar.radio("Document Type", options=["Text", "Code"])
if type == "Text":
    text_splitter_cls = st.sidebar.selectbox(
        "Text Splitter",
        options=[
            "TokenTextSplitter",
            "SentenceSplitter",
            "LC:RecursiveCharacterTextSplitter",
            "LC:CharacterTextSplitter",
        ],
    )

    chunk_size = st.sidebar.slider("Chunk Size", value=512, min_value=1, max_value=4096)
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", value=0, min_value=0, max_value=4096
    )

    if text_splitter_cls == "TokenTextSplitter":
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif text_splitter_cls == "SentenceSplitter":
        text_splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif text_splitter_cls == "LC:RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif text_splitter_cls == "LC:CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError("Unknown text splitter")
elif type == "Code":
    text_splitter_cls = st.sidebar.selectbox("Text Splitter", options=["CodeSplitter"])
    if text_splitter_cls == "CodeSplitter":
        language = st.sidebar.text_input("Language", value="python")
        max_chars = st.sidebar.slider("Max Chars", value=1500)

        text_splitter = CodeSplitter(language=language, max_chars=max_chars)
    else:
        raise ValueError("Unknown text splitter")

chunks = text_splitter.split_text(text)
tokenizer = tiktoken.get_encoding("gpt2").encode

for ind, chunk in enumerate(chunks):
    n_tokens = len(tokenizer(chunk))
    n_chars = len(chunk)
    st.text_area(f"Chunk {ind} - {n_tokens} tokens - {n_chars} chars", chunk)
