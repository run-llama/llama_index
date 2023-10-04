from llama_index.llm_predictor.base import BaseLLMPredictor, LLMPredictor
from llama_index.llm_predictor.mock import MockLLMPredictor
from llama_index.llm_predictor.structured import StructuredLLMPredictor
from llama_index.llm_predictor.vellum.predictor import VellumPredictor


def load_predictor(data: dict) -> BaseLLMPredictor:
    """Load predictor by class name."""
    if isinstance(data, BaseLLMPredictor):
        return data
    predictor_name = data.get("class_name", None)
    if predictor_name is None:
        raise ValueError("Predictor loading requires a class_name")

    if predictor_name == LLMPredictor.class_name():
        return LLMPredictor.from_dict(data)
    elif predictor_name == StructuredLLMPredictor.class_name():
        return StructuredLLMPredictor.from_dict(data)
    elif predictor_name == MockLLMPredictor.class_name():
        return MockLLMPredictor.from_dict(data)
    elif predictor_name == VellumPredictor.class_name():
        return VellumPredictor.from_dict(data)
    else:
        raise ValueError(f"Invalid predictor name: {predictor_name}")
