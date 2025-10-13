from fastapi import APIRouter, HTTPException, status, File, UploadFile
import logging
from typing import List, Dict, Any

# services
from src.service.image_processing.face_encoding_service import FaceEncodingService
from src.service.image_processing.image_processing_service import ImageProcessingService
from src.service.image_processing.multiple_image_processing import (
    MultiImageRegistrationService,
)

router = APIRouter()

encoding_service = FaceEncodingService()
image_service = ImageProcessingService()
registration_service = MultiImageRegistrationService(
    encoding_service=encoding_service, image_service=image_service
)


@router.post(
    "/v1/extract-multiple-face-encodings",
    tags=["Multi-Image Registration"],
    summary="Extract, validate, and average face encodings from multiple uploaded images.",
)
async def extract_multiple_encodings(
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    """
    Extract and validate face encodings from multiple images for user registration.
    """
    try:
        result = await registration_service.process_multi_encodings(files)

        return result

    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(
            f"Unexpected error during multi-file extraction: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}",
        )
