# GalaxEye Offline Land-Use Classification Service

A small, offline service that classifies satellite image tiles by land-use
type using a locally-run model, stores the results, and lets an analyst query
them — with no internet access and no hosted model APIs.

## Overview

This project implements the core working slice of that system: a FastAPI
service exposing `POST /classify` (classify a tile and store the result) and
`GET /predictions/{id}` (retrieve a stored result). Classification uses a
**ResNet18** with a frozen ImageNet-pretrained backbone and a trained 7-class
linear head, loaded once from a local model artifact and run on CPU. Results
are persisted to a local **SQLite** database.

The take-home deliberately keeps the slice small: more elaborate analyst
querying, authentication, monitoring, queues, and raw-image storage are out of
scope.

## Architecture

```
POST /classify (multipart tile)                    GET /predictions/{id}
        │                                                 ▲
        ▼                                                 │
   preprocess  →  classifier  →  domain result  →   SQLite  →  JSON
   (decode,64×64,  (frozen ResNet18  (id, status,    (store /  response
    normalize)      + 7-class head)   timestamp)      retrieve)
```

## Repository structure

```
src/galaxeye/
  api.py          FastAPI routes (thin orchestration)
  app.py          wiring / entrypoint (uvicorn galaxeye.app:app)
  classifier.py   ResNet18Classifier (frozen backbone + linear head)
  preprocess.py   image decoding + normalization
  storage.py      repository interface + SQLite implementation
  model.py        domain types (LandClass, Prediction, ClassifyResult, ...)
  config.py       central configuration
scripts/
  train_linear_head.py   train the 7-class linear head
  evaluate.py            evaluate the model on the held-out eval set
tests/                   pytest test suite
models/                  local model artifact
docs/
  DESIGN_NOTE.md         design note (assignment Part 1)
  CODEBASE_GUIDE.md      internal developer study guide
```

## Requirements

Python 3 with the following packages (CPU-only PyTorch is sufficient):

- `fastapi`, `uvicorn`, `python-multipart`, `httpx` (API + test client)
- `numpy`, `Pillow` (preprocessing)
- `torch`, `torchvision` (model; CPU build)
- `pytest` (tests)

## Setup

Create a virtual environment and install dependencies, e.g.:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install fastapi uvicorn python-multipart httpx numpy Pillow pytest
# CPU-only PyTorch (matches this project):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Ensure the local model artifact exists at `models/resnet18-7cls-v0.1.0.pt`. If
it is not present, train it (see [Training](#training)); training downloads the
pretrained ImageNet backbone once, then saves the local artifact used by
inference.

## Dataset

The assignment-provided dataset is intentionally **not committed** to Git
(only the read-only source data is expected; scripts reference it via
`config.py`). To run training or evaluation, place the dataset such that the
following paths resolve:

- `be-mlsys-assignment-dataset/candidate_tiles/{class}/...` — training tiles
- `be-mlsys-assignment-dataset/eval_set/...` — held-out eval tiles
- `be-mlsys-assignment-dataset/eval_labels.csv` — eval ground truth

These locations are defined in `src/galaxeye/config.py`.

## Running the service

From the repository root, with `src` on the Python path:

```bash
set PYTHONPATH=src        # Windows
# export PYTHONPATH=src   # macOS/Linux

python -m uvicorn galaxeye.app:app --host 127.0.0.1 --port 8000
```

The service is then available at `http://127.0.0.1:8000` (interactive docs at
`/docs`).

## API

### POST /classify

Classify a tile and store the result.

**Request:** `multipart/form-data`
- `file` (required): the image file (PNG/JPEG).
- `tile_id` (optional): a caller-supplied identifier; one is generated if
  omitted.

```bash
curl -X POST -F "file=@tile.png" \
     -F "tile_id=my-tile-001" \
     http://127.0.0.1:8000/classify
```

**Response (`200`):**

```json
{
  "id": "<unique result id>",
  "tile_id": "my-tile-001",
  "prediction": "Forest",
  "confidence": 0.9779,
  "status": "CONFIDENT",
  "created_at": "<UTC ISO-8601 timestamp>",
  "model_version": "resnet18-7cls-v0.1.0"
}
```

Fields:
- `id` — unique result ID (UUID).
- `tile_id` — the caller-supplied value, or a generated one.
- `prediction` — predicted land-use class.
- `confidence` — softmax probability of the predicted class (0–1).
- `status` — `CONFIDENT` or `LOW_CONFIDENCE`, per the configured threshold.
- `created_at` — UTC timestamp.
- `model_version` — the model that produced this result.

**Errors:**
- `400` — the uploaded file could not be decoded as a valid image.
- `422` — the request is missing or otherwise invalid (e.g. no `file`).
- `500` — server/storage/unexpected error.

### GET /predictions/{id}

Retrieve a previously stored result by its result ID.

```bash
curl http://127.0.0.1:8000/predictions/<id>
```

Returns the same JSON shape as `POST /classify`; returns `404` if no result
exists for that ID.

## Model

- **ResNet18** with an ImageNet-pretrained backbone.
- The backbone is **frozen**; only a **7-class linear head** is trained on the
  candidate tiles.
- Weights are saved to a **local artifact** (`models/resnet18-7cls-v0.1.0.pt`)
  and loaded at startup.
- Inference runs on **CPU, offline**, and never downloads weights or calls a
  hosted model API.

## Training

Train the 7-class linear head on `candidate_tiles`:

```bash
set PYTHONPATH=src
python -m scripts.train_linear_head
```

This freezes the pretrained backbone, replaces the head with a 7-class linear
layer, and trains the head (Adam, batch size 32, 20 epochs) before saving the
artifact.

## Evaluation

Score the model on the held-out eval set (requires the eval dataset):

```bash
set PYTHONPATH=src
python -m scripts.evaluate
```

Actual held-out results (210 tiles):

```
overall accuracy: 0.8714  (183/210 correct)
correct: 183     incorrect: 27
```

Per-class (precision / recall / F1):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AnnualCrop | 0.9032 | 0.9333 | 0.9180 |
| Forest | 0.8485 | 0.9333 | 0.8889 |
| Highway | 0.7667 | 0.7667 | 0.7667 |
| Industrial | 0.8182 | 0.9000 | 0.8571 |
| Residential | 0.9062 | 0.9667 | 0.9355 |
| River | 0.8846 | 0.7667 | 0.8214 |
| SeaLake | 1.0000 | 0.8333 | 0.9091 |

Confusion matrix (rows = actual, columns = predicted):

```
             AnnualCro  Forest  Highway  Industria Residenti  River  SeaLake
AnnualCrop        28       0       2         0         0       0       0
Forest             0      28       1         0         1       0       0
Highway            2       0      23         3         0       2       0
Industrial         0       0       1        27         2       0       0
Residential        0       0       0         1        29       0       0
River              1       1       3         2         0      23       0
SeaLake            0       4       0         0         0       1      25
```

Notes:
- The eval set was **held out**; the model was **not** modified based on eval
  results.
- Evaluation ran **offline**.
- **27/210** tiles were misclassified.
- **87.14%** is held-out **evaluation** accuracy, not training accuracy.

## Testing

```bash
python -m pytest -q
```

Current verified result: **38 passed**, with 1 pre-existing deprecation warning
(from the FastAPI test client).

## Offline guarantee

Runtime inference requires no internet: the model artifact is bundled locally,
the classifier loads the local `state_dict` (no pretrained-weights download at
inference), and no hosted model API is used.

## Design / deeper documentation

- `docs/DESIGN_NOTE.md` — the design note covering architecture, trade-offs,
  assumptions, and open questions.
- `docs/CODEBASE_GUIDE.md` — an **internal/developer study guide** (a learning
  aid for modifying the codebase), not part of the formal assignment
  deliverable.

## Limitations / future improvements

Honest limitations of the current implementation:

- **SQLite** — single-writer; limited concurrent write throughput.
- **Single-process architecture** — no horizontal scaling.
- **No authentication** — the service is not access-controlled.
- **No monitoring/alerting** — nothing surfaces drift, latency, or failures.
- **Limited querying** — only `GET /predictions/{id}` is exposed; the
  repository's richer query support is not routed.
- **Confidence is not calibrated** — the softmax score is a policy input, not a
  true probability.
- **No raw image retention** — only prediction metadata is stored.
