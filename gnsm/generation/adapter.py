"""Optional trainable projection for soft-prompt or cross-attention injection."""

from __future__ import annotations

from gnsm.exceptions import OptionalDependencyError

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None


if nn is not None:

    class StatePrefixAdapter(nn.Module):
        def __init__(self, state_dim: int, model_dim: int, prefix_tokens: int = 8) -> None:
            super().__init__()
            self.prefix_tokens = prefix_tokens
            self.model_dim = model_dim
            self.projection = nn.Sequential(
                nn.Linear(state_dim, model_dim * 2),
                nn.GELU(),
                nn.Linear(model_dim * 2, prefix_tokens * model_dim),
            )

        def forward(self, state: Tensor) -> Tensor:
            prefix = self.projection(state)
            return prefix.view(state.shape[0], self.prefix_tokens, self.model_dim)

    class StateCrossAttentionAdapter(nn.Module):
        def __init__(self, state_dim: int, model_dim: int, heads: int = 8) -> None:
            super().__init__()
            self.key_value_projection = nn.Linear(state_dim, model_dim)
            self.cross_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
            self.gate = nn.Parameter(torch.tensor(-2.0))

        def forward(self, hidden: Tensor, state_tokens: Tensor) -> Tensor:
            key_values = self.key_value_projection(state_tokens)
            attended, _ = self.cross_attention(hidden, key_values, key_values, need_weights=False)
            return hidden + torch.sigmoid(self.gate) * attended


else:

    class _MissingTorch:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise OptionalDependencyError(
                "Conditioning adapters require `python -m pip install -e '.[training]'`."
            )

    StatePrefixAdapter = _MissingTorch
    StateCrossAttentionAdapter = _MissingTorch
