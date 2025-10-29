"""
Noisemaker: Procedural noise generation with Python and TensorFlow.

This package provides tools for generating and composing procedural noise patterns,
applying image effects, and creating generative art.
"""

from noisemaker.composer import Preset
from noisemaker.constants import (
    ColorSpace,
    DistanceMetric,
    InterpolationType,
    OctaveBlending,
    PointDistribution,
    ValueDistribution,
    ValueMask,
    VoronoiDiagramType,
    WormBehavior,
)
from noisemaker.generators import multires, basic

__version__ = "0.5.0"

__all__ = [
    # Core API
    "Preset",
    "multires",
    "basic",
    # Constants/Enums
    "ColorSpace",
    "DistanceMetric",
    "InterpolationType",
    "OctaveBlending",
    "PointDistribution",
    "ValueDistribution",
    "ValueMask",
    "VoronoiDiagramType",
    "WormBehavior",
]
