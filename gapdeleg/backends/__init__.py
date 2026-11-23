from .base import CloudModel, EdgeModel, Generation, Message, UnguardedEgressError, Usage


def make_edge(cfg: dict):
    cfg = dict(cfg)
    kind = cfg.pop("backend")
    if kind == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend(**cfg)
    if kind == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend(**cfg)
    if kind == "llama_cpp":
        from .llama_cpp_backend import LlamaCppBackend
        return LlamaCppBackend(**cfg)
    raise ValueError(f"unknown edge backend {kind!r}")


def make_cloud(cfg: dict, cache_dir=None) -> CloudModel:
    cfg = dict(cfg)
    kind = cfg.pop("backend")
    cfg.setdefault("cache_dir", cache_dir)
    if kind == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(**cfg)
    if kind == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(**cfg)
    if kind == "ollama":
        from .ollama_cloud import OllamaCloud
        return OllamaCloud(**cfg)
    raise ValueError(f"unknown cloud backend {kind!r}")


__all__ = ["CloudModel", "EdgeModel", "Generation", "Message", "UnguardedEgressError", "Usage",
           "make_cloud", "make_edge"]
