import logging
from typing import Dict, List

import numpy as np
from fastapi import HTTPException, UploadFile

from src.service.image_processing.face_encoding_comparing_service import (
    FaceEncodingService,
)

# Import the services that handle the atomic tasks
from src.service.image_processing.image_processing_service import ImageProcessingService

logging.basicConfig(level=logging.WARNING)


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
        # 1. Minimum File Count Validation (This could stay in the router but is handled here for end-to-end logic)
        if len(files) < 5:
            raise HTTPException(
                status_code=400,
                detail="At least 5 images required for registration",
            )

        sum_encoding = np.zeros(128, dtype=np.float32)
        count = 0

        for file in files:
            encoding = await self._image_service.extract_single_encoding(file)
            sum_encoding += encoding
            count += 1

        averaged_encoding = (sum_encoding / count).tolist()

        # 4. Final Calculation
        # Delegate averaging and scoring to the dedicated Encoding Service
        averaged_encoding_list = self._encoding_service.average_encodings(
            averaged_encoding
        )

        # 5. Return structured result
        return {
            "success": True,
            "message": "Face encodings validated and averaged",
            "facialEncoding": averaged_encoding_list,
        }
