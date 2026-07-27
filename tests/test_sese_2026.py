import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_sese_2026 import (  # noqa: E402
    _components,
    _ordered_runtime_map,
    _parse_devices,
    _source_adjacency,
    _source_clusters,
    _source_entropy_variants,
)


def test_parse_devices_preserves_single_device_compatibility_and_accepts_lists():
    assert _parse_devices("cuda:7", None) == ("cuda:7",)
    assert _parse_devices("cuda:7", "cuda:0,cuda:2 cuda:5") == (
        "cuda:0",
        "cuda:2",
        "cuda:5",
    )
    with pytest.raises(ValueError, match="duplicate SeSE devices"):
        _parse_devices("cuda:0", "cuda:0,cuda:0")


def test_ordered_runtime_map_keeps_input_order_when_workers_finish_out_of_order():
    barrier = threading.Barrier(3)

    def evaluate(item, runtime):
        if item < 3:
            barrier.wait(timeout=1)
            time.sleep((2 - item) * 0.01)
        return item, runtime

    observed = list(
        _ordered_runtime_map(list(range(6)), ["gpu0", "gpu1", "gpu2"], evaluate)
    )
    assert [item for item, _ in observed] == list(range(6))
    assert {runtime for _, runtime in observed} <= {"gpu0", "gpu1", "gpu2"}


def test_source_clusters_follow_released_hybrid_threshold():
    cosine = np.asarray(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    entailment = np.asarray(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    labels = _source_clusters(cosine, entailment)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_source_adjacency_repairs_clusters_with_maximum_spanning_edges():
    entailment = np.asarray(
        [
            [0.0, 0.8, 0.3, 0.2],
            [0.8, 0.0, 0.4, 0.1],
            [0.3, 0.4, 0.0, 0.9],
            [0.2, 0.1, 0.9, 0.0],
        ]
    )
    adjacency = _source_adjacency(entailment, [0, 0, 1, 1])
    assert np.allclose(adjacency, adjacency.T)
    assert len(_components(adjacency)) == 1
    assert np.count_nonzero(np.triu(adjacency, 1)) == 3
    assert np.isclose(adjacency[0, 1], 0.8)
    assert np.isclose(adjacency[2, 3], 0.9)


def test_source_entropy_variants_keep_raw_and_name_sign_correction():
    variants = _source_entropy_variants(-3.5)
    assert variants == {
        "SeSE-source-raw": -3.5,
        "SeSE-source-sign-corrected": 3.5,
    }
