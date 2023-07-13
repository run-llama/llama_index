"""
A wrapper to handle multi-process embeddings. When the server is inactive for `inactive_wind_down`
seconds, the models will be unloaded from memory.
"""

class MultiProcessEmbeddingsWrapper:
    def __init__(