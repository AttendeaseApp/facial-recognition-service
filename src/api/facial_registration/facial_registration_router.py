import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.service.image_processing.image_processing_service import ImageProcessingService

router = APIRouter()
logger = logging.getLogger(__name__)

image_service = ImageProcessingService()


class FaceEncodingResponse(BaseModel):
    """Response model for single face encoding extraction."""

    success: bool
    facialEncoding: List[float]
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post(
    "/extract-face-encoding",
    tags=["Face Verification/Extraction"],
    summary="Extract face encoding from a single image for verification",
    response_model=FaceEncodingResponse,
)
async def extract_face_encoding_for_verification(
    file: UploadFile = File(..., description="Single facial image for verification"),
) -> FaceEncodingResponse:
    """
    Extract facial encoding from a single uploaded image.
    Used during authentication/verification to compare against stored encoding.

    This endpoint:
    - Accepts 1 image
    - Validates image has one clear face
    - Checks image quality
    - Returns encoding for comparison

    Returns:
        - success: Boolean indicating operation success
        - facialEncoding: 128-dimensional face encoding
        - message: Human-readable status message
        - metadata: Quality score and processing details
    """
    try:
        logger.info(f"Extracting face encoding from file: {file.filename}")
        result = await image_service.extract_multiple_encodings(
            files=[file], required_count=1
        )

        quality = result["metadata"]["average_quality"]
        message = "Face encoding extracted successfully"

        if quality < 50:
            logger.warning(f"Low quality image ({quality:.2f}) for verification")
            message = f"Face detected but image quality is low ({quality:.1f}/100). Consider retaking in better lighting for more accurate verification."

        return FaceEncodingResponse(
            success=True,
            facialEncoding=result["facialEncoding"],
            message=message,
            metadata=result["metadata"],
        )

    except HTTPException as e:
        logger.warning(f"Client error during face encoding extraction: {e.detail}")
        raise e

    except Exception as e:
        logger.error(
            f"Unexpected error during face encoding extraction: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}",
        )
