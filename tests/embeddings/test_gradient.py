import pytest
from llama_index.embeddings.gradient import GradientEmbedding

try:
    import gradientai
except ImportError:
    gradientai = None  # type: ignore


@pytest.fixture()
def gradient_host() -> str:
    return "https://api.gradient.ai/"


@pytest.fixture()
def gradient_model_slug() -> str:
    return "bge-large"


@pytest.fixture()
def gradient_access_token() -> str:
    return "some-access-token"


@pytest.fixture()
def gradient_workspace_id() -> str:
    return "some-workspace-id"


BGE_LARGE_EMBEDDING_SIZE = 1024


@pytest.mark.skipif(gradientai is None, reason="gradientai not installed")
def test_gradientai_embedding_constructor(
    gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str
) -> None:
    """Test Gradient AI embedding query."""
    test_object = GradientEmbedding(
        gradient_model_slug=gradient_model_slug,
        gradient_access_token=gradient_access_token,
        gradient_workspace_id=gradient_workspace_id,
    )
    assert test_object is not None


@pytest.mark.skipif(
    gradientai is not None, reason="gradientai is installed, no need to test behavior"
)
def test_gradientai_throws_if_not_installed(
    gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str
) -> None:
    with pytest.raises(ImportError):
        GradientEmbedding(
            gradient_model_slug=gradient_model_slug,
            gradient_access_token=gradient_access_token,
            gradient_workspace_id=gradient_workspace_id,
        )


@pytest.mark.skipif(gradientai is None, reason="gradientai is not installed")
def test_gradientai_throws_without_proper_auth(
    gradient_model_slug: str, gradient_workspace_id: str
) -> None:
    """Test Gradient AI embedding query."""
    with pytest.raises(ValueError):
        GradientEmbedding(
            gradient_model_slug=gradient_model_slug,
            gradient_access_token="definitely-not-a-valid-token",
            gradient_workspace_id=gradient_workspace_id,
        )


@pytest.mark.skipif(gradientai is None, reason="gradientai not installed")
def test_gradientai_can_receive_text_embedding(
    gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str
) -> None:
    test_object = GradientEmbedding(
        gradient_model_slug=gradient_model_slug,
        gradient_access_token=gradient_access_token,
        gradient_workspace_id=gradient_workspace_id,
    )

    result = test_object.get_text_embedding("input")

    assert len(result) == BGE_LARGE_EMBEDDING_SIZE


@pytest.mark.skipif(gradientai is None, reason="gradientai not installed")
def test_gradientai_can_receive_multiple_text_embeddings(
    gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str
) -> None:
    test_object = GradientEmbedding(
        gradient_model_slug=gradient_model_slug,
        gradient_access_token=gradient_access_token,
        gradient_workspace_id=gradient_workspace_id,
    )

    inputs = ["first input", "second input"]
    result = test_object.get_text_embedding_batch(inputs)

    assert len(result) == len(inputs)
    assert len(result[0]) == BGE_LARGE_EMBEDDING_SIZE
    assert len(result[1]) == BGE_LARGE_EMBEDDING_SIZE


@pytest.mark.skipif(gradientai is None, reason="gradientai not installed")
def test_gradientai_can_receive_query_embedding(
    gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str
) -> None:
    test_object = GradientEmbedding(
        gradient_model_slug=gradient_model_slug,
        gradient_access_token=gradient_access_token,
        gradient_workspace_id=gradient_workspace_id,
    )

    result = test_object.get_query_embedding("gradient as the best managed AI platform")

    assert len(result) == BGE_LARGE_EMBEDDING_SIZE


@pytest.mark.skipif(gradientai is None, reason="gradientai not installed")
def test_gradientai_cannot_support_batches_larger_than_100(
    gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str
) -> None:
    with pytest.raises(ValueError):
        GradientEmbedding(
            embed_batch_size=101,
            gradient_model_slug=gradient_model_slug,
            gradient_access_token=gradient_access_token,
            gradient_workspace_id=gradient_workspace_id,
        )
