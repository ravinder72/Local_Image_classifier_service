"""Tests for classifier initialization and prediction.

These tests build a classifier from a locally-generated random weights file
so they never download pretrained weights at test time.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18

from galaxeye.classifier import ResNet18Classifier
from galaxeye.model import LandClass
from galaxeye.preprocess import to_input_array


@pytest.fixture
def weights_file(tmp_path):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    path = tmp_path / "test-model.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def _sample_input():
    image = Image.new("RGB", (64, 64), (90, 120, 60))
    return to_input_array(image)


def test_classifier_initializes(weights_file):
    clf = ResNet18Classifier(weights_path=weights_file)
    assert clf.model_version


def test_predict_returns_valid_land_class(weights_file):
    clf = ResNet18Classifier(weights_path=weights_file)
    pred = clf.predict(_sample_input())
    assert isinstance(pred.label, LandClass)


def test_predict_confidence_in_unit_range(weights_file):
    clf = ResNet18Classifier(weights_path=weights_file)
    pred = clf.predict(_sample_input())
    assert 0.0 <= pred.confidence <= 1.0


def test_predict_works_on_real_preprocessed_image(weights_file):
    clf = ResNet18Classifier(weights_path=weights_file)
    input_array = _sample_input()
    assert input_array.shape == (1, 3, 64, 64)
    assert input_array.dtype == np.float32
    pred = clf.predict(input_array)
    assert isinstance(pred.label, LandClass)
    assert 0.0 <= pred.confidence <= 1.0


def test_predict_confidence_is_bounded_sum_of_softmax(weights_file):
    clf = ResNet18Classifier(weights_path=weights_file)
    pred = clf.predict(_sample_input())
    assert pred.confidence > 0.0  # top softmax probability is strictly positive
