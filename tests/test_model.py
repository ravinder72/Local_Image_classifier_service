"""Tests for the domain/result model."""

from dataclasses import fields

import pytest

from galaxeye.model import ClassifyResult, LandClass, Prediction, ResultStatus


def test_land_class_values_match_dataset_folders():
    assert {c.value for c in LandClass} == {
        "AnnualCrop",
        "Forest",
        "Highway",
        "Industrial",
        "Residential",
        "River",
        "SeaLake",
    }


def test_prediction_has_only_label_and_confidence():
    field_names = {f.name for f in fields(Prediction)}
    assert field_names == {"label", "confidence"}


def test_prediction_is_frozen():
    p = Prediction(label=LandClass.FOREST, confidence=0.9)
    with pytest.raises(Exception):
        p.confidence = 0.5


def test_result_status_values():
    assert {s.value for s in ResultStatus} == {"CONFIDENT", "LOW_CONFIDENCE"}


def test_classify_result_round_trip_fields():
    result = ClassifyResult(
        id="abc",
        tile_id="tile_001",
        label=LandClass.RIVER,
        confidence=0.42,
        status=ResultStatus.LOW_CONFIDENCE,
        created_at="2026-09-24T12:00:00Z",
        model_version="resnet18-7cls-v0.1.0",
    )
    assert result.id == "abc"
    assert result.tile_id == "tile_001"
    assert result.label is LandClass.RIVER
    assert result.status is ResultStatus.LOW_CONFIDENCE
    assert result.model_version == "resnet18-7cls-v0.1.0"
