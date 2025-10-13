import numpy as np
from typing import List, Dict
from fastapi import HTTPException, UploadFile
import logging

# Import the services that handle the atomic tasks
from src.service.image_processing.image_processing_service import ImageProcessingService
from src.service.image_processing.face_encoding_service import FaceEncodingService

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
        # 1. Minimum File Count Validation (This could stay in the router but is handled here for end-to-end logic)
        if len(files) < 3:
            # Re-using the logic from the original code
            raise HTTPException(
                status_code=400,
                detail="At least 3 images required for registration",
            )

        encodings: List[np.ndarray] = []

        # 2. Extraction Loop
        for file in files:
            # Delegate single image extraction and validation to the dedicated Image Service
            # This service method raises HTTPException if it fails (e.g., no face, multiple faces)
            encoding = await self._image_service.extract_single_encoding(file)
            encodings.append(encoding)

        # 3. Consistency Check
        # Delegate consistency validation to the dedicated Encoding Service
        # This service method raises HTTPException if validation fails
        self._encoding_service.validate_encoding_consistency(encodings, threshold=0.6)

        # 4. Final Calculation
        # Delegate averaging and scoring to the dedicated Encoding Service
        averaged_encoding_list = self._encoding_service.average_encodings(encodings)
        consistency_score = self._encoding_service.calculate_consistency_score(
            encodings
        )

        # 5. Return structured result
        return {
            "success": True,
            "message": "Face encodings validated and averaged",
            "facialEncoding": averaged_encoding_list,
            "confidence_score": consistency_score,
        }
