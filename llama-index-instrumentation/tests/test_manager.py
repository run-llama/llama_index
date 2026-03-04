import llama_index_instrumentation as instrument


def test_root_manager_add_dispatcher():
    # arrange
    root_manager = instrument.root_manager

    # act
    dispatcher = instrument.get_dispatcher("test")

    # assert
    assert "root" in root_manager.dispatchers
    assert "test" in root_manager.dispatchers
