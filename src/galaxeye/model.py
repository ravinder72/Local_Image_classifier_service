"""Domain/result model.

Pure data types with no I/O or framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LandClass(Enum):
    """The 7 land-use classes. Values match the dataset folder names."""

    ANNUAL_CROP = "AnnualCrop"
    FOREST = "Forest"
    HIGHWAY = "Highway"
    INDUSTRIAL = "Industrial"
    RESIDENTIAL = "Residential"
    RIVER = "River"
    SEA_LAKE = "SeaLake"


class ResultStatus(Enum):
    """Policy flag describing how confident a prediction is."""

    CONFIDENT = "CONFIDENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True)
class Prediction:
    """Output of the classifier.

    Contains only label and confidence. The classifier may use logits
    internally, but they are not part of the public contract.
    """

    label: LandClass
    confidence: float


@dataclass(frozen=True)
class ClassifyResult:
    """A persisted classification record (the analyst-queryable unit)."""

    id: str
    tile_id: str
    label: LandClass
    confidence: float
    status: ResultStatus
    created_at: str
    model_version: str
