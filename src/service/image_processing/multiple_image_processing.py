import logging
from typing import Dict, List

import numpy as np
from fastapi import HTTPException, UploadFile

from src.service.image_processing.face_encoding_comparing_service import (
    FaceEncodingService,
)
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
        if len(files) < 5:
            raise HTTPException(
                status_code=400,
                detail="At least 5 images required for registration",
            )

        sum_encoding = np.zeros(128, dtype=np.float32)
        encodings_count = 0
        first_encoding = None

        for file in files:
            encoding = await self._image_service.extract_single_encoding(file)

            if first_encoding is None:
                first_encoding = encoding
            else:
                distance = np.linalg.norm(encoding - first_encoding)
                if distance > 0.6:
                    raise HTTPException(
                        status_code=400,
                        detail="Inconsistent facial encodings detected across images.",
                    )

            sum_encoding += encoding
            encodings_count += 1

        average_encoding = (sum_encoding / encodings_count).tolist()

        return {
            "success": True,
            "message": "Face encodings validated and averaged",
            "facialEncoding": average_encoding,
        }
