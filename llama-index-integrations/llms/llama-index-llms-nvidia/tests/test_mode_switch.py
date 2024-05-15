import pytest

from llama_index.llms.nvidia import NVIDIA as Interface
from llama_index.llms.nvidia.base import BASE_URL


def test_mode_switch_nvidia_throws_without_key(masked_env_var: str):
    x = Interface()
    with pytest.raises(ValueError):
        x.mode("nvidia")


def test_mode_switch_nvidia_with_key(masked_env_var: str):
    Interface().mode("nvidia", api_key="test")


def test_mode_switch_nim_throws_without_url():
    instance = Interface()
    with pytest.raises(ValueError):
        instance.mode("nim")


def test_mode_switch_nim_with_url():
    Interface().mode("nim", base_url="test")


def test_mode_switch_param_setting():
    instance = Interface(model="dummy")

    instance1 = instance.mode("nim", base_url="https://test_url/v1/")
    assert instance1.model == "dummy"
    assert str(instance1.api_base) == "https://test_url/v1/"

    instance2 = instance1.mode("nvidia", api_key="test", model="dummy-2")
    assert instance2.model == "dummy-2"
    assert str(instance2.api_base) == BASE_URL
    assert instance2.api_key == "test"
