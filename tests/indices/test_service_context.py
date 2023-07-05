from llama_index.indices.service_context import ServiceContext


def test_global_service_context() -> None:
    # check initialized
    orig_ctx = ServiceContext.get_global()
    assert orig_ctx is not None

    # Test setting
    global_ctx = ServiceContext.from_defaults().set_global()
    get_global = ServiceContext.get_global()
    assert get_global is not None
    assert get_global.llm_predictor is not None

    # Test mutation
    global_ctx.llm_predictor = None  # type: ignore
    get_global = ServiceContext.get_global()
    assert get_global is not None
    assert get_global.llm_predictor is None

    # Test setting to none
    ServiceContext.set_global_to_none()
    assert ServiceContext.get_global() is None

    # Reset
    orig_ctx.set_global()
