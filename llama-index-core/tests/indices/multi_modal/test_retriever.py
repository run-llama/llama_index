from llama_index.core import Document, MockEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import (
    ImageDocument,
    IndexNode,
    NodeWithScore,
    QueryBundle,
)

BASE64_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="


def _retriever(**kwargs):
    index = MultiModalVectorStoreIndex.from_documents(
        [Document.example(), ImageDocument(image=BASE64_PNG)],
        image_embed_model=MockMultiModalEmbedding(embed_dim=3),
        embed_model=MockEmbedding(embed_dim=3),
    )
    return index.as_retriever(similarity_top_k=2, image_similarity_top_k=1, **kwargs)


def test_retriever_initializes_base_retriever_state():
    # Regression test for #22080: the constructor used to skip
    # BaseRetriever.__init__, leaving object_map/_verbose unset.
    retriever = _retriever()
    assert retriever.object_map == {}
    assert retriever._verbose is False
    assert retriever.callback_manager is not None


def test_retriever_keeps_explicit_callback_manager_and_object_map():
    callback_manager = CallbackManager()
    retriever = _retriever(callback_manager=callback_manager, object_map={"idx": "obj"})
    assert retriever.callback_manager is callback_manager
    assert retriever.object_map == {"idx": "obj"}


def test_retrieve_path_with_index_node_does_not_raise():
    # retrieve() always runs _handle_recursive_retrieval, which reads
    # self.object_map for IndexNode results; it used to raise
    # AttributeError: 'MultiModalVectorIndexRetriever' object has no
    # attribute 'object_map'.
    retriever = _retriever()
    nodes = retriever._handle_recursive_retrieval(
        QueryBundle("hello"),
        [NodeWithScore(node=IndexNode(index_id="inner-obj", text="inner"), score=1.0)],
    )
    assert len(nodes) == 1
    assert nodes[0].node.index_id == "inner-obj"
