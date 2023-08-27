"""Test simple graph store index."""

import pytest
import sys

import json
import logging
import os

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import fsspec
from dataclasses_json import DataClassJsonMixin

from llama_index.graph_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    GraphStore,
)


# who will test the testers?
def test_one_false():
    assert 1 == 0
