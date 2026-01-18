import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.service.image_processing.image_processing_service import ImageProcessingService

router = APIRouter()
logger = logging.getLogger(__name__)
image_service = ImageProcessingService()


class FaceEncodingResponse(BaseModel):
    """Response model for face encoding extraction."""

    success: bool
    facialEncoding: List[float]
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post(
    "/extract-multiple-face-encodings",
    tags=["Face Registration"],
    summary="Extract averaged face encoding from 5 images for registration",
    response_model=FaceEncodingResponse,
)
async def extract_multiple_face_encodings(
    files: List[UploadFile] = File(..., description="5 facial images for registration"),
) -> FaceEncodingResponse:
    """
    Extract averaged facial encoding from 5 uploaded images.
    Used during registration to create a robust face profile.

    This endpoint:
    - Accepts exactly 5 images
    - Validates each image has one clear face
    - Checks image quality for all images
    - Returns averaged encoding for better accuracy

    Args:
        files: List of exactly 5 image files (sent as multipart/form-data with field name "files")

    Returns:
        - success: Boolean indicating operation success
        - facialEncoding: 128-dimensional averaged face encoding
        - message: Human-readable status message
        - metadata: Quality scores and processing details for all images

    Raises:
        400: Invalid number of images, no face detected, multiple faces, or poor quality
        413: Request size exceeds limit
        422: Validation error (wrong field name or format)
        500: Server processing error
    """
    try:
        # Log request details
        logger.info(f"Received {len(files)} files for face encoding extraction")

        # Log individual file sizes
        for idx, file in enumerate(files, 1):
            file_size_kb = len(await file.read()) / 1024
            await file.seek(0)  # Reset file pointer
            logger.info(
                f"Processing image {idx}: {file.filename} ({file_size_kb:.0f} KB)"
            )

        # Calculate total size
        total_size = sum([len(await f.read()) for f in files])
        for f in files:
            await f.seek(0)  # Reset all file pointers
        logger.info(
            f"Total upload size: {total_size / 1024 / 1024:.0f} MB ({total_size} bytes)"
        )

        # Validate file count
        if len(files) != 5:
            logger.warning(
                f"Invalid number of files: expected 5, received {len(files)}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected 5 images, received {len(files)}. Please upload exactly 5 clear photos of your face.",
            )

        logger.info("Sending request to facial service...")

        # Process images through the service
        result = await image_service.extract_multiple_encodings(
            files=files, required_count=5
        )

        quality = result["metadata"]["average_quality"]
        quality_scores = result["metadata"]["quality_scores"]

        # Generate appropriate message based on quality
        if quality >= 70:
            message = "Face encodings extracted successfully with excellent quality"
        elif quality >= 50:
            message = f"Face encodings extracted successfully (average quality: {quality:.1f}/100)"
        else:
            message = (
                f"Faces detected but average image quality is low ({quality:.1f}/100). "
                "Consider retaking photos in better lighting for more accurate registration."
            )
            logger.warning(f"Low average quality ({quality:.2f}) across images")

        # Log individual quality scores
        for idx, score in enumerate(quality_scores, 1):
            logger.info(f"Image {idx} quality score: {score:.2f}/100")

        logger.info(
            f"Successfully processed {len(files)} images with average quality {quality:.2f}"
        )

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
            f"Unexpected error during face encoding extraction: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}",
        )
