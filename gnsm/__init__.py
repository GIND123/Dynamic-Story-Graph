"""Grounded Narrative State Model.

The public package stays dependency-light. Heavy training integrations are
loaded only from their specific modules.
"""

from gnsm.pipeline import GNSMSystem, SceneResult
from gnsm.schemas import GraphDelta, NarrativeState, PlotAction, SceneGraph

__all__ = [
    "GNSMSystem",
    "GraphDelta",
    "NarrativeState",
    "PlotAction",
    "SceneGraph",
    "SceneResult",
]

__version__ = "0.1.0"
