from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse
import logging

from src.data.face_verification_request_model import FaceVerificationRequest

# services
from src.service.image_processing.face_encoding_service import (
    FaceEncodingService,
)

router = APIRouter()
service = FaceEncodingService()


@router.post(
    "/verification/authenticate-face",
    tags=["Face Verification"],
    summary="Authenticate a user by comparing two face encodings.",
)
async def authenticate_face(request: FaceVerificationRequest):
    """
    Compare two facial encodings to verify if they match.
    """
    try:
        result = service.compare_encodings(
            uploaded_encoding=request.uploaded_encoding,
            reference_encoding=request.reference_encoding,
        )

        return {
            "success": True,
            "verified": True,
            **result,
        }

    except Exception as e:
        detail = str(e.detail) if isinstance(e, HTTPException) else str(e)
        status_code = (
            e.status_code
            if isinstance(e, HTTPException)
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )

        logging.error(f"Error during face authentication: {detail}", exc_info=True)

        return JSONResponse(
            status_code=status_code,
            content={"success": False, "error": detail},
        )
