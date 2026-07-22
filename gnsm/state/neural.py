"""Optional PyTorch Graph Transformer skeleton for the learned C4/C5 modules."""

from __future__ import annotations

from gnsm.exceptions import OptionalDependencyError

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised only without training extras
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None


if nn is not None:

    class GraphStateEncoder(nn.Module):
        """Graph-transformer encoder with node and global narrative outputs."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 768,
            layers: int = 4,
            heads: int = 8,
            edge_types: int = 8,
        ) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.edge_embedding = nn.Embedding(edge_types, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            self.graph_layers = nn.TransformerEncoder(layer, num_layers=layers)
            self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.output_norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            node_features: Tensor,
            padding_mask: Tensor | None = None,
        ) -> tuple[Tensor, Tensor]:
            batch_size = node_features.shape[0]
            nodes = self.input_projection(node_features)
            global_token = self.global_token.expand(batch_size, -1, -1)
            sequence = torch.cat([global_token, nodes], dim=1)
            if padding_mask is not None:
                global_mask = torch.zeros(
                    (batch_size, 1), dtype=torch.bool, device=padding_mask.device
                )
                padding_mask = torch.cat([global_mask, padding_mask], dim=1)
            encoded = self.graph_layers(sequence, src_key_padding_mask=padding_mask)
            encoded = self.output_norm(encoded)
            return encoded[:, 1:], encoded[:, 0]

    class GroundedStateHeads(nn.Module):
        """Non-negotiable graph reconstruction plus auxiliary grounded heads."""

        def __init__(
            self,
            hidden_dim: int,
            edge_types: int,
            attribute_classes: int,
            emotion_classes: int,
        ) -> None:
            super().__init__()
            self.edge_decoder = nn.Bilinear(hidden_dim, hidden_dim, edge_types)
            self.attribute_decoder = nn.Linear(hidden_dim, attribute_classes)
            self.emotion_decoder = nn.Linear(hidden_dim, emotion_classes)

        def forward(
            self,
            source: Tensor,
            target: Tensor,
            global_state: Tensor,
        ) -> dict[str, Tensor]:
            return {
                "edge_logits": self.edge_decoder(source, target),
                "attribute_logits": self.attribute_decoder(source),
                "emotion_logits": self.emotion_decoder(global_state),
            }

    class NeuralTransitionModel(nn.Module):
        """Predict next latent state and supervised graph-delta labels."""

        def __init__(self, hidden_dim: int, delta_classes: int) -> None:
            super().__init__()
            self.transition = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.delta_head = nn.Linear(hidden_dim, delta_classes)

        def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
            next_state = self.transition(torch.cat([state, action], dim=-1))
            return next_state, self.delta_head(next_state)


else:

    class _MissingTorch:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise OptionalDependencyError(
                "PyTorch modules require `python -m pip install -e '.[training]'`."
            )

    GraphStateEncoder = _MissingTorch
    GroundedStateHeads = _MissingTorch
    NeuralTransitionModel = _MissingTorch
