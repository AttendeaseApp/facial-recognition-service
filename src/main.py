from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)

# api
from src.api.validate_facial_encoding_api import router as validate_face_encoding_router
from src.api.facial_verification.compare_face_encoding_router import (
    router as authenticate_face_router,
)
from src.api.facial_verification.extract_face_encoding_router import (
    router as authenticate_user_face,
)

from src.api.facial_registration.facial_registration_router import (
    router as register_facial_encodings,
)


app = FastAPI(
    title="Facial Recognition Service API",
    description="Facial recognition service api for encoding validation and face encoding comparison",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(validate_face_encoding_router, tags=["Face Recognition"])

app.include_router(authenticate_face_router, tags=["Face Verification"])

app.include_router(authenticate_user_face, tags=["Face Verification/Extraction"])

app.include_router(register_facial_encodings, tags=["Multi-Image Registration"])
