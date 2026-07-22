"""Integration boundary for Maverick-coref or BookNLP.

Implement `extract` against the chosen library and return the package-level
`EntityExtraction` contract. Keeping third-party objects out of the rest of the
system makes BookNLP/Maverick an experiment configuration choice.
"""

from __future__ import annotations

from gnsm.exceptions import OptionalDependencyError
from gnsm.extraction.base import EntityExtraction
from gnsm.schemas import Entity


class MaverickCoreferenceExtractor:
    def __init__(self, model_name: str = "sapienzanlp/maverick-mes-ontonotes") -> None:
        self.model_name = model_name

    def extract(self, text: str, registry: dict[str, Entity] | None = None) -> EntityExtraction:
        del text, registry
        raise OptionalDependencyError(
            "Maverick integration is an adapter seam. Install the selected coreference "
            "backend and implement its version-specific loader in gnsm/extraction/maverick.py."
        )
