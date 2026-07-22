from gnsm.extraction import ExtractionPipeline
from gnsm.schemas import EdgeType


def test_reference_extraction_builds_grounded_graph() -> None:
    graph = ExtractionPipeline.reference().extract(
        'Mara is alive. Mara waited in the Observatory. Mara said, "Listen."',
        "scene-0",
    )
    names = {entity.name for entity in graph.entities.values()}
    assert "Mara" in names
    assert "The" not in names
    assert "Listen" not in names
    assert any(
        attribute.key == "status" and attribute.value == "alive" for attribute in graph.attributes
    )
    location_edge = next(edge for edge in graph.edges if edge.type is EdgeType.LOCATED_AT)
    assert graph.entities[location_edge.target].name == "Observatory"
    assert graph.quotes[0].speaker_id == "character:mara"
    assert graph.validate() == []


def test_registry_preserves_entity_identity_across_scenes() -> None:
    pipeline = ExtractionPipeline.reference()
    first = pipeline.extract("Mara entered.", "scene-0")
    second = pipeline.extract("Mara returned.", "scene-1", first.entities)
    assert second.mentions[0].entity_id == first.mentions[0].entity_id
