from dsg.llm.base import Backend, CachedBackend, GenerationCache
from dsg.llm.mock import MockBackend

__all__ = ["Backend", "CachedBackend", "GenerationCache", "MockBackend", "build_backend"]


def build_backend(name: str, cache_path=None):
    """``mock`` or an MLX model id / shorthand (``qwen3b``, ``qwen7b``, ``qwen1.5b``)."""
    shorthand = {
        "qwen1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "qwen3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "qwen7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    }
    if name == "mock":
        inner = MockBackend()
    else:
        from dsg.llm.mlx_backend import MLXBackend

        inner = MLXBackend(shorthand.get(name, name))
    if cache_path is None:
        return inner
    return CachedBackend(inner, GenerationCache(cache_path))
