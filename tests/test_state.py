import numpy as np

from gnsm.extraction import ExtractionPipeline
from gnsm.state import HashingStateEncoder, collapse_diagnostics
from gnsm.state.probes import LinearProbe


def test_hashing_encoder_is_stable_and_sized() -> None:
    graph = ExtractionPipeline.reference().extract("Mara is alive. Ivo is missing.", "s0")
    encoder = HashingStateEncoder(dimension=32)
    first = encoder.encode(graph)
    second = encoder.encode(graph)
    assert first.dimension == 32
    np.testing.assert_array_equal(first.global_vector, second.global_vector)
    assert set(first.node_vectors) == set(graph.entities)


def test_collapse_diagnostics_detect_constant_embeddings() -> None:
    diagnostics = collapse_diagnostics(np.ones((5, 4), dtype=np.float32))
    assert diagnostics["effective_rank"] == 0.0
    assert diagnostics["mean_variance"] == 0.0


def test_linear_probe_decodes_separable_state() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array(["left", "left", "right", "right"])
    probe = LinearProbe().fit(features, labels)
    assert probe.score(features, labels) == 1.0
