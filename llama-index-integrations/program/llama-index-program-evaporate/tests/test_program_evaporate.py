from llama_index.core.program import BasePydanticProgram
from llama_index.program.evaporate import DFEvaporateProgram


def test_class():
    names_of_base_classes = [b.__name__ for b in DFEvaporateProgram.__mro__]
    assert BasePydanticProgram.__name__ in names_of_base_classes
