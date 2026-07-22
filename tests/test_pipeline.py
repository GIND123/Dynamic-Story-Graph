from gnsm.pipeline import GNSMSystem
from gnsm.schemas import PlotAction


def test_reference_pipeline_runs_end_to_end() -> None:
    system = GNSMSystem.reference(dimension=24)
    scene = "Mara is alive. Mara waited in the Observatory."
    state = system.initialize(scene)
    result = system.generate_next(
        previous_scene=scene,
        state=state,
        action=PlotAction(
            "Mara studies the signal.",
            participants=("character:mara",),
        ),
        next_scene_id="scene-1",
    )
    assert result.verification.accepted
    assert result.attempts == 1
    assert result.state.dimension == 24
    assert "Mara" in result.text
