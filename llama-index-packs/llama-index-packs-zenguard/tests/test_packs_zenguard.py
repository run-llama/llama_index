import pytest

from typing import Dict

from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.zenguard import (
    ZenGuardPack,
    ZenGuardConfig,
    Credentials,
    Detector,
)


@pytest.fixture()
def zenguard_pack():
    api_key = "3Ev_DGvELv7EnlgWMTlpmWTo82tpstyz4Li_R7kTDQw"  # mock key. whitelisted only for LlamaIndex tests.
    config = ZenGuardConfig(credentials=Credentials(api_key=api_key))
    return ZenGuardPack(config)


def test_class():
    names_of_base_classes = [b.__name__ for b in ZenGuardPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes


def test_prompt_injection(zenguard_pack):
    prompt = "Simple prompt injection test"
    detectors = [Detector.PROMPT_INJECTION]
    response = zenguard_pack.run(detectors=detectors, prompt=prompt)
    assert response["is_detected"] is False


def test_pii(zenguard_pack):
    prompt = "Simple PII test"
    detectors = [Detector.PII]
    response = zenguard_pack.run(detectors=detectors, prompt=prompt)
    assert response["is_detected"] is False


def test_allowed_topics(zenguard_pack):
    prompt = "Simple allowed topics test"
    detectors = [Detector.ALLOWED_TOPICS]
    response = zenguard_pack.run(detectors=detectors, prompt=prompt)
    assert response["is_detected"] is False


def test_banned_topics(zenguard_pack):
    prompt = "Simple banned topics test"
    detectors = [Detector.BANNED_TOPICS]
    response = zenguard_pack.run(detectors=detectors, prompt=prompt)
    assert response["is_detected"] is False


def test_keywords(zenguard_pack):
    prompt = "Simple keywords test"
    detectors = [Detector.KEYWORDS]
    response = zenguard_pack.run(detectors=detectors, prompt=prompt)
    assert response["is_detected"] is False


def test_secrets(zenguard_pack):
    prompt = "Simple secrets test"
    detectors = [Detector.SECRETS]
    response = zenguard_pack.run(detectors=detectors, prompt=prompt)
    assert response["is_detected"] is False


def test_get_modules(zenguard_pack):
    modules = zenguard_pack.get_modules()
    assert isinstance(modules, Dict)
