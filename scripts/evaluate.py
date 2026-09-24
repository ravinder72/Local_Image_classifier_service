"""Evaluate the existing classifier on the held-out eval_set.

Read-only: uses the existing local model artifact and the exact same
preprocessing/inference path as the API (ResNet18Classifier +
preprocess.decode_image/to_input_array + LandClass mapping). It never
downloads anything, never uses a hosted model, and never modifies the model,
training procedure, or preprocessing.

The eval_set is held out for evaluation only.

Usage:
    PYTHONPATH=src python -m scripts.evaluate
"""

from __future__ import annotations

import csv

from PIL import Image

from galaxeye.classifier import ResNet18Classifier
from galaxeye.config import EVAL_LABELS_CSV, EVAL_SET_DIR
from galaxeye.model import LandClass
from galaxeye.preprocess import decode_image, to_input_array

CLASSES = list(LandClass)


def load_ground_truth() -> dict[str, LandClass]:
    labels: dict[str, LandClass] = {}
    with open(EVAL_LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = LandClass(row["true_label"])
    return labels


def evaluate() -> None:
    classifier = ResNet18Classifier()
    ground_truth = load_ground_truth()
    filenames = sorted(p.name for p in EVAL_SET_DIR.glob("*.png"))

    predictions: list[tuple[str, LandClass, LandClass, float]] = []
    for filename in filenames:
        if filename not in ground_truth:
            continue
        data = (EVAL_SET_DIR / filename).read_bytes()
        input_array = to_input_array(decode_image(data))  # same as the API
        pred = classifier.predict(input_array)
        predictions.append(
            (filename, ground_truth[filename], pred.label, pred.confidence)
        )

    total = len(predictions)
    correct = sum(1 for _, actual, predicted, _ in predictions if actual == predicted)
    incorrect = total - correct
    accuracy = correct / total if total else 0.0

    # Per-class counts: TP, FP, FN
    tp = {c: 0 for c in CLASSES}
    fp = {c: 0 for c in CLASSES}
    fn = {c: 0 for c in CLASSES}
    for _, actual, predicted, _ in predictions:
        if actual == predicted:
            tp[actual] += 1
        else:
            fp[predicted] += 1
            fn[actual] += 1

    print("=" * 64)
    print(f"EVALUATION on eval_set  ({total} tiles)")
    print("=" * 64)
    print(f"overall accuracy : {accuracy:.4f}  ({correct}/{total})")
    print(f"correct          : {correct}")
    print(f"incorrect        : {incorrect}")
    print("-" * 64)
    print(f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}")
    print("-" * 64)
    for c in CLASSES:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{c.value:<14}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}")
    print("-" * 64)

    print("confusion matrix (rows=actual, cols=predicted):")
    header = "            " + "".join(f"{c.value[:9]:>10}" for c in CLASSES)
    print(header)
    matrix = {a: {p: 0 for p in CLASSES} for a in CLASSES}
    for _, actual, predicted, _ in predictions:
        matrix[actual][predicted] += 1
    for a in CLASSES:
        row = f"{a.value:<12}"
        for p in CLASSES:
            row += f"{matrix[a][p]:>10}"
        print(row)
    print("-" * 64)

    print("misclassified tiles:")
    for filename, actual, predicted, conf in predictions:
        if actual != predicted:
            print(f"  {filename:<12} actual={actual.value:<12} "
                  f"predicted={predicted.value:<12} confidence={conf:.4f}")


if __name__ == "__main__":
    evaluate()
