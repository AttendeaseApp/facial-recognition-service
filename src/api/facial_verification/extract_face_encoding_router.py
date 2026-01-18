from typing import Any, Dict

from fastapi import APIRouter, File, UploadFile

# services
from src.service.image_processing.image_processing_service import (
    ImageProcessingService,
)

router = APIRouter()
base64_extractor_service = ImageProcessingService()


@router.post("/extract-face-encoding")
async def extract_face_encoding_for_verification(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Extract face encoding from a single image for verification.
    Used during authentication to compare against stored encoding.
    """
    service = ImageProcessingService()

    result = await service.extract_multiple_encodings(files=[file], required_count=1)

    return {
        "success": True,
        "facialEncoding": result["facialEncoding"],
        "quality": result["metadata"]["average_quality"],
    }
