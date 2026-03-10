"""
Knowledge Capsule Lifecycle for LlamaIndex
Based on Memory-Like-A-Tree
"""

from enum import Enum
from typing import Dict, Optional
import time

class Phase(Enum):
    SPROUT = "sprout"
    GREEN = "green_leaf"
    YELLOW = "yellow_leaf" 
    RED = "red_leaf"
    SOIL = "soil"

class CapsuleLifecycle:
    def __init__(self):
        self.capsules: Dict[str, dict] = {}
    
    def add(self, cid: str, content: str, priority: str = "P2"):
        self.capsules[cid] = {
            'content': content,
            'priority': priority,
            'confidence': 0.7,
            'phase': Phase.SPROUT,
            'created': time.time(),
            'accessed': time.time()
        }
    
    def access(self, cid: str) -> bool:
        if cid not in self.capsules:
            return False
        c = self.capsules[cid]
        c['accessed'] = time.time()
        c['confidence'] = min(1.0, c['confidence'] + 0.03)
        c['phase'] = Phase.GREEN if c['confidence'] >= 0.8 else Phase.SPROUT
        return True
    
    def decay(self):
        for c in self.capsules.values():
            decay = {'P0': 0, 'P1': 0.004, 'P2': 0.008}.get(c['priority'], 0.008)
            c['confidence'] = max(0, c['confidence'] - decay)
