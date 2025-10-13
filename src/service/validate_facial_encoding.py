import numpy as np
from fastapi import HTTPException, status
import logging
from typing import List

logging.basicConfig(level=logging.INFO)


class FaceEncodingService:
    """
    Service layer for facial encoding operations and validation.
    """

    def validate_and_format_encoding(
        self, face_encoding_input: List[float]
    ) -> List[float]:
        """
        Validates and formats the facial encoding input.

        Args:
            face_encoding_input: The facial encoding data, expected as a list of floats.

        Returns:
            The validated and formatted facial encoding as a list of floats.

        Raises:
            HTTPException: If the encoding is invalid (wrong type, length, or contains non-finite values).
        """
        if not isinstance(face_encoding_input, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input facialEncoding must be a list of floats.",
            )

        try:
            face_encoding = np.array(face_encoding_input, dtype=np.float64)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input facialEncoding contains non-numeric values.",
            )

        if len(face_encoding) != 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid face encoding length. Expected a 128-dimensional vector.",
            )

        if not np.all(np.isfinite(face_encoding)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Face encoding contains invalid values (e.g., NaN or infinite).",
            )

        logging.info("Face encoding validated successfully.")

        return face_encoding.tolist()
