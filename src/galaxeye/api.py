"""HTTP API layer.

Thin orchestration only: receive the upload, run the approved flow
(decode -> preprocess -> classify -> build result -> persist -> return),
and map exceptions to HTTP status codes. No preprocessing, inference, or
SQL lives directly in these routes.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from galaxeye.config import CONFIDENCE_THRESHOLD
from galaxeye.model import ClassifyResult, ResultStatus
from galaxeye.preprocess import ImageDecodeError, decode_image, to_input_array
from galaxeye.storage import ResultRepository, StorageError

logger = logging.getLogger(__name__)


def create_app(
    classifier,
    repository: ResultRepository,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> FastAPI:
    """Build the FastAPI app from injected classifier and repository."""

    app = FastAPI(title="GalaxEye Land-Use Classifier", version="0.1.0")

    @app.exception_handler(Exception)
    async def unhandled_error(request, exc) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.post("/classify")
    async def classify(
        file: UploadFile = File(...),
        tile_id: str | None = Form(default=None),
    ) -> dict:
        try:
            data = await file.read()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Could not read file") from exc

        try:
            input_array = to_input_array(decode_image(data))
        except ImageDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        prediction = classifier.predict(input_array)
        result = _build_result(
            prediction=prediction,
            tile_id=tile_id or _generate_tile_id(),
            model_version=classifier.model_version,
            confidence_threshold=confidence_threshold,
        )

        try:
            repository.save(result)
        except StorageError as exc:
            logger.exception("Storage failure saving result %s", result.id)
            raise HTTPException(status_code=500, detail="Failed to store result") from exc

        return _to_dict(result)

    @app.get("/predictions/{result_id}")
    async def get_prediction(result_id: str) -> dict:
        try:
            result = repository.get(result_id)
        except StorageError as exc:
            logger.exception("Storage failure fetching %s", result_id)
            raise HTTPException(status_code=500, detail="Failed to fetch result") from exc
        if result is None:
            raise HTTPException(status_code=404, detail="Result not found")
        return _to_dict(result)

    return app


def _build_result(prediction, tile_id: str, model_version: str, confidence_threshold: float) -> ClassifyResult:
    status = (
        ResultStatus.CONFIDENT
        if prediction.confidence >= confidence_threshold
        else ResultStatus.LOW_CONFIDENCE
    )
    return ClassifyResult(
        id=uuid.uuid4().hex,
        tile_id=tile_id,
        label=prediction.label,
        confidence=prediction.confidence,
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_version=model_version,
    )


def _generate_tile_id() -> str:
    return f"tile_{uuid.uuid4().hex[:12]}"


def _to_dict(result: ClassifyResult) -> dict:
    return {
        "id": result.id,
        "tile_id": result.tile_id,
        "prediction": result.label.value,
        "confidence": result.confidence,
        "status": result.status.value,
        "created_at": result.created_at,
        "model_version": result.model_version,
    }
