import pytest
from llama_index.legacy.llms.vllm import Vllm

try:
    vllm_init = True
except ImportError:
    vllm_init = False


@pytest.mark.skipif(vllm_init is True, reason="vertex not installed")
def test_vllm_initialization() -> None:
    llm = Vllm()
    assert llm.class_name() == "Vllm"


@pytest.mark.skipif(vllm_init is True, reason="vertex not installed")
def test_vllm_call() -> None:
    llm = Vllm(temperature=0)
    output = llm.complete("Say foo:")
    assert isinstance(output.text, str)
