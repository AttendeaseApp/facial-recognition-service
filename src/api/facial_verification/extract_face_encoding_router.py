import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

# models
from src.data.face_image_request_model import FaceImageRequest

# services
from src.service.image_processing.image_processing_service import (
    ImageProcessingService,
)

router = APIRouter()
base64_extractor_service = ImageProcessingService()


@router.post(
    "/extract-face-encoding",
    tags=["Face Verification/Extraction"],
    summary="Extract a single face encoding from a Base64 image string.",
)
async def extract_face_encoding_base64(request: FaceImageRequest):
    """
    Extracts and validates a single 128-dimensional face encoding from a Base64 image string.
    Returns the encoding as a list of floating-point numbers.
    """
    try:
        encoding_list = base64_extractor_service.extract_encoding_from_base64(
            image_base64=request.image_base64
        )

        return {
            "success": True,
            "facialEncoding": encoding_list,
        }

    except HTTPException as e:
        logging.error(
            f"HTTPException during Base64 extraction: {e.detail}", exc_info=False
        )
        return JSONResponse(
            status_code=e.status_code, content={"success": False, "error": e.detail}
        )
    except Exception as e:
        logging.error(
            f"Unexpected server error during Base64 extraction: {e}", exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": f"An unexpected server error occurred: {str(e)}",
            },
        )
