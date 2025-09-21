"""Comprehensive cross-language parity tests for the Voronoi effect.

Each test invokes the corresponding JavaScript implementation in a subprocess
and compares the raw tensor output directly with the Python version. The goal
is to exercise as many permutations of the effect as possible so that we can
identify line-by-line differences between the two implementations.
"""

import numpy as np
import pytest

from noisemaker import generators, rng, value
from noisemaker.constants import (
    DistanceMetric,
    PointDistribution,
    VoronoiDiagramType,
)

from .seeds import PARITY_SEEDS
from .utils import js_voronoi

# Five randomly chosen 32-bit seeds shared across parity tests.
SEEDS = PARITY_SEEDS

DIAGRAM_TYPES = [
    VoronoiDiagramType.range,
    VoronoiDiagramType.color_range,
    VoronoiDiagramType.flow,
    VoronoiDiagramType.color_flow,
]

DIST_METRICS = DistanceMetric.all()

POINT_DISTRIBS = [
    PointDistribution.random,
    PointDistribution.square,
    PointDistribution.waffle,
    PointDistribution.chess,
    PointDistribution.h_hex,
    PointDistribution.v_hex,
    PointDistribution.spiral,
    PointDistribution.circular,
    PointDistribution.concentric,
    PointDistribution.rotating,
]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("diagram_type", DIAGRAM_TYPES)
@pytest.mark.parametrize("dist_metric", DIST_METRICS)
def test_voronoi_diagram_and_metric(seed, diagram_type, dist_metric):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=diagram_type,
        dist_metric=dist_metric,
    )
    params = {
        "seed": seed,
        "diagram_type": diagram_type.value,
        "nth": 0,
        "dist_metric": dist_metric.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": False,
        "ridges_hint": False,
        "refract_y_from_offset": True,
        "point_freq": 3,
        "point_generations": 1,
        "point_distrib": PointDistribution.random.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    if diagram_type == VoronoiDiagramType.flow:
        atol = 5e-5
    elif diagram_type == VoronoiDiagramType.color_flow:
        atol = 3e-2
    else:
        atol = 2e-6
    assert np.allclose(tensor.numpy(), js, atol=atol)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("nth", [-1, 0, 1])
def test_voronoi_nth(seed, nth):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.range,
        dist_metric=DistanceMetric.euclidean,
        nth=nth,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.range.value,
        "nth": nth,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": False,
        "ridges_hint": False,
        "refract_y_from_offset": True,
        "point_freq": 3,
        "point_generations": 1,
        "point_distrib": PointDistribution.random.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=2e-6)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("point_distrib", POINT_DISTRIBS)
def test_voronoi_point_distrib(seed, point_distrib):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.range,
        dist_metric=DistanceMetric.euclidean,
        point_distrib=point_distrib,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.range.value,
        "nth": 0,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": False,
        "ridges_hint": False,
        "refract_y_from_offset": True,
        "point_freq": 3,
        "point_generations": 1,
        "point_distrib": point_distrib.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=2e-6)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("with_refract,refract_y_from_offset", [(0.5, True), (0.5, False)])
def test_voronoi_with_refract(seed, with_refract, refract_y_from_offset):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.range,
        dist_metric=DistanceMetric.euclidean,
        with_refract=with_refract,
        refract_y_from_offset=refract_y_from_offset,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.range.value,
        "nth": 0,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": with_refract,
        "inverse": False,
        "ridges_hint": False,
        "refract_y_from_offset": refract_y_from_offset,
        "point_freq": 3,
        "point_generations": 1,
        "point_distrib": PointDistribution.random.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=7e-3)


@pytest.mark.parametrize("seed", SEEDS)
def test_voronoi_inverse(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.range,
        dist_metric=DistanceMetric.euclidean,
        inverse=True,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.range.value,
        "nth": 0,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": True,
        "ridges_hint": False,
        "refract_y_from_offset": True,
        "point_freq": 3,
        "point_generations": 1,
        "point_distrib": PointDistribution.random.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=2e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_voronoi_ridges_hint(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.color_range,
        dist_metric=DistanceMetric.euclidean,
        ridges_hint=True,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.color_range.value,
        "nth": 0,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": False,
        "ridges_hint": True,
        "refract_y_from_offset": True,
        "point_freq": 3,
        "point_generations": 1,
        "point_distrib": PointDistribution.random.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=2e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_voronoi_point_freq_one(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.range,
        dist_metric=DistanceMetric.euclidean,
        point_freq=1,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.range.value,
        "nth": 0,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": False,
        "ridges_hint": False,
        "refract_y_from_offset": True,
        "point_freq": 1,
        "point_generations": 1,
        "point_distrib": PointDistribution.square.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=2e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_voronoi_point_generations_drift_corners(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.range,
        dist_metric=DistanceMetric.euclidean,
        point_generations=2,
        point_drift=0.5,
        point_corners=True,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.range.value,
        "nth": 0,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": False,
        "ridges_hint": False,
        "refract_y_from_offset": True,
        "point_freq": 3,
        "point_generations": 2,
        "point_distrib": PointDistribution.random.value,
        "point_drift": 0.5,
        "point_corners": True,
        "downsample": True,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=2e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_voronoi_downsample_false(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = value.voronoi(
        tensor,
        [128, 128, 3],
        diagram_type=VoronoiDiagramType.range,
        dist_metric=DistanceMetric.euclidean,
        downsample=False,
    )
    params = {
        "seed": seed,
        "diagram_type": VoronoiDiagramType.range.value,
        "nth": 0,
        "dist_metric": DistanceMetric.euclidean.value,
        "sdf_sides": 3,
        "alpha": 1.0,
        "with_refract": 0.0,
        "inverse": False,
        "ridges_hint": False,
        "refract_y_from_offset": True,
        "point_freq": 3,
        "point_generations": 1,
        "point_distrib": PointDistribution.random.value,
        "point_drift": 0.0,
        "point_corners": False,
        "downsample": False,
    }
    js = js_voronoi(params)
    assert tensor.shape == (128, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=2e-6)
