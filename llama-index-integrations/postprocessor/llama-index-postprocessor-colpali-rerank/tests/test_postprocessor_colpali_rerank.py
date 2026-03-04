import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.colpali_rerank import ColPaliRerank
from PIL import Image


def test_class():
    names_of_base_classes = [b.__name__ for b in ColPaliRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


@patch("llama_index.postprocessor.colpali_rerank.base.ColPali")
def test_init(mock_model):
    # Setup mock returns
    mock_model.from_pretrained.return_value = MagicMock()

    m = ColPaliRerank(top_n=10)

    assert m.model == "vidore/colpali-v1.2"
    assert m.top_n == 10

    # Verify the model was initialized with correct parameters
    mock_model.from_pretrained.assert_called_once()


@patch("llama_index.postprocessor.colpali_rerank.base.ColPali")
def test_postprocess(mock_colpali):
    # Create temporary image files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create two blank images
        image1_path = os.path.join(temp_dir, "image1.png")
        image2_path = os.path.join(temp_dir, "image2.png")

        white_square = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image = Image.fromarray(white_square)
        image.save(image1_path)
        image.save(image2_path)

        # Create mock nodes
        node1 = NodeWithScore(
            node=TextNode(text="test1", metadata={"file_path": image1_path}), score=0.8
        )
        node2 = NodeWithScore(
            node=TextNode(text="test2", metadata={"file_path": image2_path}), score=0.6
        )
        nodes = [node1, node2]

        # Mock the ColPali model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_colpali.from_pretrained.return_value = mock_model

        # Mock the similarity scores
        mock_processor.score_multi_vector.return_value = torch.tensor([[0.9, 0.7]])

        # Initialize the reranker
        reranker = ColPaliRerank(top_n=2, keep_retrieval_score=True)
        reranker._processor = mock_processor
        reranker._model = mock_model

        # Create a query bundle
        query = QueryBundle(query_str="test query")

        # Process the nodes
        reranked_nodes = reranker._postprocess_nodes(nodes, query)

        # Assertions
        assert len(reranked_nodes) == 2
        assert reranked_nodes[0].score > reranked_nodes[1].score
        assert reranked_nodes[0].node.metadata["retrieval_score"] == 0.8
        assert reranked_nodes[1].node.metadata["retrieval_score"] == 0.6
