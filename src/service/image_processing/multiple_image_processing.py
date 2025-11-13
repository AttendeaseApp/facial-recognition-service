import logging
from typing import Dict, List

import numpy as np
from fastapi import HTTPException, UploadFile

from src.service.image_processing.face_encoding_comparing_service import (
    FaceEncodingService,
)
from src.service.image_processing.image_processing_service import ImageProcessingService

logging.basicConfig(level=logging.INFO)


class MultiImageRegistrationService:
    """
    Service layer for coordinating the extraction, validation, and averaging of
    face encodings from multiple image files for registration.
    """

    def __init__(
        self,
        encoding_service: FaceEncodingService,
        image_service: ImageProcessingService,
    ):
        """Inject dependencies."""
        self._encoding_service = encoding_service
        self._image_service = image_service

    async def process_multi_encodings(self, files: List[UploadFile]) -> Dict:
        """
        Extracts, validates consistency, and averages face encodings from files.
        """
        if len(files) < 5:
            raise HTTPException(
                status_code=400,
                detail="At least 5 images required for registration",
            )

        encodings: List[np.ndarray] = []

        for file in files:
            encoding = await self._image_service.extract_single_encoding(file)
            encodings.append(encoding)

        self._encoding_service.validate_encoding_consistency(encodings, threshold=0.6)

        averaged_encoding_list = self._encoding_service.average_encodings(encodings)

        return {
            "success": True,
            "message": "Face encodings validated and averaged",
            "facialEncoding": averaged_encoding_list,
        }
