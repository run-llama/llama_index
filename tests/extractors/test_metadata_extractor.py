"""Test dataset generation."""

import tempfile
import typing
import urllib.request

from llama_index import SimpleDirectoryReader
from llama_index.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter


def two_random_integers(range_limit: int) -> typing.Tuple[int, int]:
    import random

    index1 = random.randint(0, range_limit - 1)
    index2 = index1
    while index2 == index1:
        index2 = random.randint(0, range_limit - 1)
    return index1, index2


def test_metadata_extractor() -> None:
    """Test metadata extraction."""
    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdirname:
        urllib.request.urlretrieve(
            "https://www.dropbox.com/scl/fi/6dlqdk6e2k1mjhi8dee5j/uber.pdf?rlkey=2jyoe49bg2vwdlz30l76czq6g&dl=1",
            f"{tmpdirname}/10k-132.pdf",
        )
        urllib.request.urlretrieve(
            "https://www.dropbox.com/scl/fi/qn7g3vrk5mqb18ko4e5in/lyft.pdf?rlkey=j6jxtjwo8zbstdo4wz3ns8zoj&dl=1",
            f"{tmpdirname}/10k-vFinal.pdf",
        )

        # Note the uninformative document file name, which may be a common scenario in a production setting
        uber_docs = SimpleDirectoryReader(
            input_files=[f"{tmpdirname}/10k-132.pdf"]
        ).load_data()
        uber_front_pages = uber_docs[0:3]
        uber_content = uber_docs[63:69]
        uber_docs = uber_front_pages + uber_content

        text_splitter = TokenTextSplitter(
            separator=" ", chunk_size=512, chunk_overlap=128
        )

        extractors = [
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),
        ]

        transformations = [text_splitter, *extractors]

        pipeline = IngestionPipeline(transformations=transformations)

        uber_nodes = pipeline.run(documents=uber_docs)

        assert len(uber_nodes) == 21
        index1, index2 = two_random_integers(len(uber_nodes))
        assert (
            uber_nodes[index1].metadata["document_title"]
            != uber_nodes[index2].metadata["document_title"]
        )
