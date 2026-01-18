from typing import Any, Dict

from fastapi import APIRouter, File, UploadFile

# services
from src.service.image_processing.image_processing_service import (
    ImageProcessingService,
)

router = APIRouter()
image_service = ImageProcessingService()


@router.post("/extract-face-encoding")
async def extract_face_encoding_for_verification(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Extract face encoding from a single image for verification.
    Used during authentication to compare against stored encoding.

    This endpoint:
    - Accepts exactly 1 image
    - Validates image has one clear face
    - Checks image quality
    - Returns encoding for comparison with stored registration

    Args:
        file: Single image file for facial verification

    Returns:
        - success: Boolean indicating operation success
        - facialEncoding: 128-dimensional face encoding
        - quality: Quality score (0-100)
        - metadata: Processing details

    Raises:
        400: No face detected, multiple faces, or poor quality
        413: File size exceeds limit
        500: Server processing error
    """
    result = await image_service.extract_single_encoding(file)

    return {
        "success": True,
        "facialEncoding": result["facialEncoding"],
        "quality": result["metadata"]["average_quality"],
        "metadata": {
            "quality_score": result["metadata"]["quality_scores"][0],
            "processing_time": result["metadata"]["processing_time"],
        },
    }
