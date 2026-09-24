# Design Note — Offline Land-Use Classification Service

## Problem / goal

GalaxEye works with satellite imagery, much of it processed **offline on
isolated hardware with no internet**. This service classifies land-use for
individual satellite tiles using a locally-run model, stores the results, and
lets an analyst query them — entirely offline, on CPU, with no hosted model
APIs.

The goal of this take-home is not a finished production system but a clean,
reasoned, working slice: one endpoint that ingests a tile, classifies it with
a local model, and persists the result, with the surrounding decisions and
trade-offs explained. Accuracy is secondary to how the system is designed and
reasoned about.

## System architecture

A single-process FastAPI service. The classifier is a ResNet18 with a frozen
ImageNet-pretrained backbone and a trained 7-class linear head, loaded once
from a local artifact. Results persist to an embedded SQLite database.

```
            POST /classify (multipart tile [+ optional tile_id])
                    │
                    ▼
        preprocess (decode → RGB → 64×64 → normalize)   [preprocess.py]
                    │  numpy (1,3,64,64) float32
                    ▼
        classifier (frozen ResNet18 + 7-class head)      [classifier.py]
                    │  Prediction(label, confidence)
                    ▼
        domain / result layer (id, tile_id, status,      [model.py, api.py]
        timestamp, model_version)
                    │
                    ▼
        SQLite repository (INSERT + commit)              [storage.py]
                    │
                    ▼
        JSON response   │   GET /predictions/{id}  →  stored result
```

**Implemented in the thin slice:** `POST /classify`, `GET /predictions/{id}`,
the preprocessing/model/persistence pipeline, model training
(`scripts/train_linear_head.py`), and evaluation (`scripts/evaluate.py`).

**Intentionally stubbed / out of scope:** analyst search/list/statistics
endpoints (the repository has `query()` support but no route exposes it),
authentication, monitoring/alerting, async queues, raw image storage.

## Tile flow

1. Tile upload (`multipart/form-data`, optional `tile_id`).
2. Image decoding — bytes validated and converted to an RGB PIL image
   (`preprocess.decode_image`).
3. Preprocessing — resize to 64×64, scale to [0,1], ImageNet normalize to a
   `(1,3,64,64)` float32 array (`preprocess.to_input_array`).
4. Local model inference — frozen backbone extracts features, the linear head
   scores the 7 classes, softmax yields probabilities.
5. Prediction + confidence — the argmax class and its softmax probability.
6. Result metadata — status (from the confidence threshold), result UUID,
   tile_id, UTC timestamp, model_version.
7. Persistence — `INSERT` into SQLite.
8. Response — the stored result serialized as JSON.

## Stored result

We store: **result ID** (unique UUID, primary key), **tile ID** (caller-facing
correlation key), **predicted class**, **confidence**, **status** (CONFIDENT /
LOW_CONFIDENCE), **created_at** (UTC), and **model_version**.

- **tile_id vs result ID:** a tile is an external entity a caller can identify;
  it can be re-classified (e.g. by a newer model version), producing multiple
  result rows for the same tile. The **result ID** uniquely identifies a single
  stored classification. Hence they are separate fields with different roles.

We deliberately store prediction **metadata only** — the raw image bytes are
**not** currently stored.

## Querying

The intended analyst capability is to query stored results (by class,
confidence, recency). The current slice implements `GET /predictions/{id}` and
the repository exposes `query(land_class, min_confidence, limit)` internally.
More elaborate analyst search / list / statistics endpoints are **not** exposed
and are intentionally out of scope for the thin slice.

## Confidence handling

- Confidence is the model's **softmax** probability for the predicted class.
- A configured threshold (`CONFIDENCE_THRESHOLD`, default 0.8) classifies each
  prediction as **CONFIDENT** or **LOW_CONFIDENCE**.
- The threshold is a **policy value**, not a calibrated probability. A softmax
  score is a probability distribution over classes, but it is **not** claimed
  to equal the true probability that the prediction is correct. Any decision
  based on it should be validated/calibrated before production use.

## Model choice

- **ResNet18** — a compact, well-understood CNN that runs well on CPU.
- **ImageNet-pretrained backbone** — strong general visual features.
- **Frozen backbone + trainable 7-class linear head** — only the head is trained on the ~1050 labelled tiles;
  this keeps training simple, reduces the number of trainable parameters, and is easy to explain and modify.
- **7 classes** — the dataset has 7 land-use classes (AnnualCrop, Forest,
  Highway, Industrial, Residential, River, SeaLake).
- **Local saved artifact** (`models/resnet18-7cls-v0.1.0.pt`) — loaded at
  runtime; **CPU inference; no hosted model API**.

## Storage choice

**SQLite** is appropriate here because it is:
- **offline** — a local file, no network or server,
- **embedded** — runs inside the process, zero additional infrastructure,
- **simple** — one file, easy to inspect and back up,
- **transactional** — atomic `INSERT`/`commit`,
- **queryable** — real SQL for analyst queries.

Trade-off: PostgreSQL (or similar) offers stronger concurrency, multi-process
writers, and richer operational features, but requires a database service —
unnecessary infrastructure for this offline, single-node take-home. We note it
as a production upgrade path.

## Offline operation

Runtime remains offline because:
- the model artifact is bundled locally,
- the classifier builds the architecture with `weights=None` and loads the
  local `state_dict` (it never downloads pretrained weights),
- no hosted inference API is called,
- there is no external infrastructure dependency.

(Pretrained ImageNet weights are fetched once at **training time** to produce
the local artifact; inference itself is fully local.)

## Trade-offs / future production evolution

| Decision | Why OK for take-home | Production change |
|---|---|---|
| Single process | Simplest; one tile is fast on CPU | Worker/queue + separate services at scale |
| SQLite | Offline, embedded, no service | Server DB (e.g. Postgres) for concurrency |
| No raw image storage | Keeps DB small; metadata-only | Object store for raw tiles if retention needed |
| No authentication | Local offline tool | AuthN/Z when exposed |
| No monitoring/alerting | In scope of the slice | Health checks, metrics, alerting for unattended ops |
| Limited query endpoints | Slice scope | Analyst list/search/statistics endpoints |
| Local model artifact | Bundled, offline, simple | Model registry + versioned rollout |
| CPU inference | Fits offline CPU constraint | GPU/accelerators if latency/scale demand |

## Assumptions / open questions

- **Throughput/concurrency:** expected request volume and concurrency?
- **Retention:** how long must results (and possibly raw tiles) be retained?
- **Raw imagery:** must the original tile bytes be retained for audit/review?
- **Latency:** acceptable end-to-end latency for a classification?
- **Confidence/calibration:** is a calibrated confidence required, and what
  downstream action depends on it?
- **Model update/versioning policy:** how often are models updated, and must
  old results remain attributable to their model version?
- **Authentication/access control:** who may classify and query?
- **Monitoring/alerting:** what must be surfaced for an unattended deployment?
