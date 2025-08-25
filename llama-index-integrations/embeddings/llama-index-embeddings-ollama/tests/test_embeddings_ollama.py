from unittest.mock import patch
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding


def test_embedding_class():
    """Test basic class instantiation."""
    emb = OllamaEmbedding(
        model_name="", client_kwargs={"headers": {"Authorization": "Bearer token"}}
    )
    assert isinstance(emb, BaseEmbedding)


class TestInstructionFunctionality:
    """Test cases for the new instruction functionality."""

    def test_instruction_fields_default_none(self):
        """Test that instruction fields default to None."""
        embedder = OllamaEmbedding(model_name="test-model")
        assert embedder.query_instruction is None
        assert embedder.text_instruction is None

    def test_instruction_fields_set_correctly(self):
        """Test that instruction fields are properly set."""
        embedder = OllamaEmbedding(
            model_name="test-model",
            query_instruction="Query instruction:",
            text_instruction="Text instruction:",
        )

        assert embedder.query_instruction == "Query instruction:"
        assert embedder.text_instruction == "Text instruction:"

    def test_format_query_with_instruction(self):
        """Test query formatting with instruction."""
        embedder = OllamaEmbedding(
            model_name="test-model",
            query_instruction="Represent the question for retrieval:",
        )

        result = embedder._format_query("What is AI?")
        expected = "Represent the question for retrieval: What is AI?"
        assert result == expected

    def test_format_query_without_instruction(self):
        """Test query formatting without instruction."""
        embedder = OllamaEmbedding(model_name="test-model")

        result = embedder._format_query("What is AI?")
        assert result == "What is AI?"

    def test_format_text_with_instruction(self):
        """Test text formatting with instruction."""
        embedder = OllamaEmbedding(
            model_name="test-model",
            text_instruction="Represent the document for retrieval:",
        )

        result = embedder._format_text("AI is a field of computer science")
        expected = (
            "Represent the document for retrieval: AI is a field of computer science"
        )
        assert result == expected

    def test_format_text_without_instruction(self):
        """Test text formatting without instruction."""
        embedder = OllamaEmbedding(model_name="test-model")

        result = embedder._format_text("AI is a field of computer science")
        assert result == "AI is a field of computer science"

    def test_instruction_stripping(self):
        """Test that whitespace is handled correctly."""
        embedder = OllamaEmbedding(
            model_name="test-model",
            query_instruction="  Query:  ",  # Extra spaces
        )

        result = embedder._format_query("  What is AI?  ")  # Extra spaces
        expected = "Query: What is AI?"  # Should be cleaned
        assert result == expected

    def test_empty_strings(self):
        """Test handling of empty strings."""
        embedder = OllamaEmbedding(model_name="test-model", query_instruction="Query:")

        result = embedder._format_query("")
        expected = "Query:"
        assert result == expected

    @patch.object(OllamaEmbedding, "get_general_text_embedding")
    def test_query_embedding_uses_instruction(self, mock_embed):
        """Test that query embedding methods use instructions."""
        embedder = OllamaEmbedding(model_name="test-model", query_instruction="Query:")

        mock_embed.return_value = [0.1, 0.2, 0.3]

        embedder._get_query_embedding("What is AI?")

        # Verify the formatting was applied
        mock_embed.assert_called_once_with("Query: What is AI?")

    @patch.object(OllamaEmbedding, "get_general_text_embedding")
    def test_text_embedding_uses_instruction(self, mock_embed):
        """Test that text embedding methods use instructions."""
        embedder = OllamaEmbedding(model_name="test-model", text_instruction="Text:")

        mock_embed.return_value = [0.1, 0.2, 0.3]

        embedder._get_text_embedding("AI is computer science")

        # Verify the formatting was applied
        mock_embed.assert_called_once_with("Text: AI is computer science")

    @patch.object(OllamaEmbedding, "aget_general_text_embedding")
    async def test_async_query_embedding_uses_instruction(self, mock_embed):
        """Test that async query embedding methods use instructions."""
        embedder = OllamaEmbedding(
            model_name="test-model", query_instruction="Async Query:"
        )

        mock_embed.return_value = [0.1, 0.2, 0.3]

        await embedder._aget_query_embedding("What is AI?")

        # Verify the formatting was applied
        mock_embed.assert_called_once_with("Async Query: What is AI?")

    @patch.object(OllamaEmbedding, "aget_general_text_embedding")
    async def test_async_text_embedding_uses_instruction(self, mock_embed):
        """Test that async text embedding methods use instructions."""
        embedder = OllamaEmbedding(
            model_name="test-model", text_instruction="Async Text:"
        )

        mock_embed.return_value = [0.1, 0.2, 0.3]

        await embedder._aget_text_embedding("AI is computer science")

        # Verify the formatting was applied
        mock_embed.assert_called_once_with("Async Text: AI is computer science")

    @patch.object(OllamaEmbedding, "get_general_text_embedding")
    def test_batch_text_embeddings_use_instruction(self, mock_embed):
        """Test that batch text embedding methods use instructions."""
        embedder = OllamaEmbedding(model_name="test-model", text_instruction="Batch:")

        mock_embed.return_value = [0.1, 0.2, 0.3]

        embedder._get_text_embeddings(["Text 1", "Text 2"])

        # Verify both calls used the instruction
        expected_calls = [(("Batch: Text 1",),), (("Batch: Text 2",),)]
        assert mock_embed.call_args_list == expected_calls

    @patch.object(OllamaEmbedding, "aget_general_text_embedding")
    async def test_async_batch_text_embeddings_use_instruction(self, mock_embed):
        """Test that async batch text embedding methods use instructions."""
        embedder = OllamaEmbedding(
            model_name="test-model", text_instruction="Async Batch:"
        )

        mock_embed.return_value = [0.1, 0.2, 0.3]

        await embedder._aget_text_embeddings(["Text 1", "Text 2"])

        # Verify both calls used the instruction
        expected_calls = [(("Async Batch: Text 1",),), (("Async Batch: Text 2",),)]
        assert mock_embed.call_args_list == expected_calls

    def test_constructor_passes_instructions_to_parent(self):
        """Test that instructions are properly accessible as attributes."""
        embedder = OllamaEmbedding(
            model_name="test-model",
            query_instruction="Query:",
            text_instruction="Text:",
        )

        # Verify instructions are accessible as attributes
        assert embedder.query_instruction == "Query:"
        assert embedder.text_instruction == "Text:"
