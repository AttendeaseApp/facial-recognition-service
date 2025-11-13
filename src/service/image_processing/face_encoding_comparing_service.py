import logging
from typing import List

import numpy as np
from fastapi import HTTPException, status

logging.basicConfig(level=logging.WARNING)


class FaceEncodingService:
    """
    Service layer for comparing facial encodings.
    """

    DEFAULT_VERIFICATION_THRESHOLD = 0.6

    def compare_encodings(
        self,
        uploaded_encoding: List[float],
        reference_encoding: List[float],
        threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
    ) -> dict:
        """
        Compares a candidate face encoding (uploaded) against a known face encoding (reference).

        Raises:
            HTTPException: If the input encodings are invalid (wrong length, non-finite values).
        """
        try:
            #  convert input lists to numpy arrays
            uploaded_np = np.array(uploaded_encoding, dtype=np.float64)
            reference_np = np.array(reference_encoding, dtype=np.float64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or both encodings contain non-numeric data.",
            )

        # structural and data validation
        if len(uploaded_np) != 128 or len(reference_np) != 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid encoding dimensions. Expected 128-dimensional vectors.",
            )
        if not np.all(np.isfinite(uploaded_np)) or not np.all(
            np.isfinite(reference_np)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Encodings contain invalid values (e.g., NaN or infinite).",
            )

        # calculate distance (L2 norm / Euclidean distance)
        distance = np.linalg.norm(uploaded_np - reference_np)
        # determine match status and confidence
        is_match = distance < threshold
        # calculate confidence
        confidence = max(0.0, 1.0 - distance)

        logging.info(
            f"Verification completed. Distance: {distance:.4f}, Match: {is_match}"
        )

        return {
            "is_face_matched": bool(is_match),
            "face_distance": float(distance),
            "confidence": float(confidence),
        }

    def calculate_consistency_score(self, encodings: List[np.ndarray]) -> float:
        """Calculate a consistency score based on the average distance between all encoding pairs."""
        if not encodings or len(encodings) < 2:
            return 1.0
        distances = []
        n = len(encodings)

        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(encodings[i] - encodings[j])
                distances.append(distance)

        avg_distance = np.mean(distances)
        consistency_score = np.exp(-avg_distance)
        return float(consistency_score)

    def validate_encoding_consistency(
        self, encodings: List[np.ndarray], threshold: float = 0.6
    ) -> None:
        """Check if all encodings are similar enough to be from the same person (raises HTTPException on failure)."""
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                distance = np.linalg.norm(encodings[i] - encodings[j])
                if distance > threshold:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Face encodings are inconsistent. Distance between file {i + 1} and {j + 1} is too large ({distance:.4f} > {threshold}). Please recapture images.",
                    )

    def average_encodings(self, encodings: List[np.ndarray]) -> List[float]:
        """Calculates the mean of multiple face encodings for robust registration."""
        averaged_encoding = np.mean(encodings, axis=0)
        return averaged_encoding.tolist()
