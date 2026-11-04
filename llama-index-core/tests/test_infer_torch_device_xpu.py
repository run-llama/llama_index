"""Regression tests for #23667: infer_torch_device() must see Intel XPU.

``torch`` is imported inside the function, so the test patches the real torch
module's backends; no XPU hardware is needed.
"""

from types import SimpleNamespace

import torch

from llama_index.core.utils import infer_torch_device


def _fake(available: bool) -> SimpleNamespace:
    return SimpleNamespace(is_available=lambda: available)


def _patch(monkeypatch, *, cuda: bool, mps: bool, xpu):
    monkeypatch.setattr(torch, "cuda", _fake(cuda), raising=False)
    monkeypatch.setattr(torch, "backends", SimpleNamespace(mps=_fake(mps)), raising=False)
    if xpu is None:
        monkeypatch.delattr(torch, "xpu", raising=False)
    else:
        monkeypatch.setattr(torch, "xpu", _fake(xpu), raising=False)


def test_xpu_is_picked_when_available(monkeypatch) -> None:
    _patch(monkeypatch, cuda=False, mps=False, xpu=True)
    assert infer_torch_device() == "xpu"


def test_cuda_still_wins_over_xpu(monkeypatch) -> None:
    _patch(monkeypatch, cuda=True, mps=False, xpu=True)
    assert infer_torch_device() == "cuda"


def test_mps_still_wins_over_xpu(monkeypatch) -> None:
    _patch(monkeypatch, cuda=False, mps=True, xpu=True)
    assert infer_torch_device() == "mps"


def test_cpu_when_nothing_is_available(monkeypatch) -> None:
    _patch(monkeypatch, cuda=False, mps=False, xpu=False)
    assert infer_torch_device() == "cpu"


def test_build_without_xpu_attribute_still_works(monkeypatch) -> None:
    """Older CPU-only builds do not expose torch.xpu at all."""
    _patch(monkeypatch, cuda=False, mps=False, xpu=None)
    assert infer_torch_device() == "cpu"