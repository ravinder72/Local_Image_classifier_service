"""Application wiring / entrypoint.

Builds the classifier (loads the local model artifact once) and the SQLite
repository, then wires them into the FastAPI app via create_app. Run with:
    uvicorn galaxeye.app:app
"""

from fastapi import FastAPI

from galaxeye.api import create_app
from galaxeye.classifier import ResNet18Classifier
from galaxeye.config import DB_PATH
from galaxeye.storage import SQLiteResultRepository


def build_app() -> FastAPI:
    classifier = ResNet18Classifier()  # loads model weights once, at startup
    repository = SQLiteResultRepository(DB_PATH)
    return create_app(classifier, repository)


app = build_app()
