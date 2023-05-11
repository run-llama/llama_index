
from llama_index.indices.vector_store.auto_retriever.schema import (
    MetadataInfo, VectorStoreInfo)


def test_to_json():
    info = VectorStoreInfo(
        content_info='collection of movie plot summaries',
        metadata_info=[
            MetadataInfo(
                name='director',
                type='str',
                description='The director of the movie',
            ),
            MetadataInfo(
                name='theme',
                type='str',
                description='The theme of the movie',
            ),
        ]
    )

    expected_json = """{"metadata_info": [{"name": "director", "type": "str", "description": "The director of the movie"}, {"name": "theme", "type": "str", "description": "The theme of the movie"}], "content_info": "collection of movie plot summaries"}"""

    assert info.to_json() == expected_json