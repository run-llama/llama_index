import pytest
import os

from typing import Dict, Any

from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.zenguard_guardrails import (
    ZenguardGuardrailsPack,
    ZenGuardConfig,
    Credentials,
    Detector,
)


@pytest.fixture
def guardrails():
    api_key = os.environ.get("ZEN_API_KEY")
    assert api_key, "ZEN_API_KEY is not set"
    config = ZenGuardConfig(credentials=Credentials(api_key=api_key))
    return ZenguardGuardrailsPack(config)


def test_class():
    names_of_base_classes = [b.__name__ for b in ZenguardGuardrailsPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes


def assert_detectors_response(response, detectors):
    assert response is not None
    for detector in detectors:
        common_response = next((
            resp["common_response"]
            for resp in response["responses"]
            if resp["detector"] == detector.value
        ))
        assert "err" not in common_response, f"API returned an error: {common_response.get('err')}"
        assert common_response.get("is_detected") is False, f"Prompt was detected: {common_response}"


def test_prompt_injection(guardrails):
    prompt = "Simple prompt injection test"
    detectors = [Detector.PROMPT_INJECTION]
    response = guardrails.run(detectors=detectors, prompt=prompt)
    assert_detectors_response(response, detectors)


def test_pii(guardrails):
    prompt = "Simple PII test"
    detectors = [Detector.PII]
    response = guardrails.run(detectors=detectors, prompt=prompt)
    assert_detectors_response(response, detectors)


def test_allowed_topics(guardrails):
    prompt = "Simple allowed topics test"
    detectors = [Detector.ALLOWED_TOPICS]
    response = guardrails.run(detectors=detectors, prompt=prompt)
    assert_detectors_response(response, detectors)


def test_banned_topics(guardrails):
    prompt = "Simple banned topics test"
    detectors = [Detector.BANNED_TOPICS]
    response = guardrails.run(detectors=detectors, prompt=prompt)
    assert_detectors_response(response, detectors)


def test_keywords(guardrails):
    prompt = "Simple keywords test"
    detectors = [Detector.KEYWORDS]
    response = guardrails.run(detectors=detectors, prompt=prompt)
    assert_detectors_response(response, detectors)


def test_secrets(guardrails):
    prompt = "Simple secrets test"
    detectors = [Detector.SECRETS]
    response = guardrails.run(detectors=detectors, prompt=prompt)
    assert_detectors_response(response, detectors)


def test_toxicity(guardrails):
    prompt = "Simple toxicity test"
    detectors = [Detector.TOXICITY]
    response = guardrails.run(detectors=detectors, prompt=prompt)
    assert_detectors_response(response, detectors)


def test_get_modules(guardrails):
    modules = guardrails.get_modules()
    assert isinstance(modules, Dict)
