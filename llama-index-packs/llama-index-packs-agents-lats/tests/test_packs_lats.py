from llama_index.core.agent.types import BaseAgentWorker
from llama_index.core.llms.mock import MockLLM
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.agents_lats import LATSAgentWorker, LATSPack


def test_worker() -> None:
    llm = MockLLM()
    worker = LATSAgentWorker([], llm=llm)
    assert isinstance(worker, BaseAgentWorker)


def test_pack() -> None:
    llm = MockLLM()
    pack = LATSPack([], llm=llm)
    assert isinstance(pack, BaseLlamaPack)
