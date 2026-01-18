import io
import logging
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessingService:
    """
    Service layer for extracting a single facial encoding from an image.
    """

    MAX_DIMENSION = 1024
    MIN_DIMENSION = 300
    JPEG_QUALITY = 95
    MIN_FACE_SIZE = (50, 50)  # FACE DETECTION TRESHOLDS

    async def extract_multiple_encodings(
        self, files: List[UploadFile], required_count: int = 5
    ) -> dict:
        """
        Extract facial encodings from multiple images with validation.

        Args:
            files: List of uploaded image files
            required_count: Expected number of images

        Returns:
            dict with success status, averaged encoding, and metadata
        """
        if len(files) != required_count:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {required_count} images, received {len(files)}",
            )

        encodings_list = []
        face_quality_scores = []

        for idx, file in enumerate(files):
            try:
                logger.info(f"Processing image {idx + 1}/{len(files)}: {file.filename}")
                encoding, quality_score = await self._process_single_image(file, idx)
                encodings_list.append(encoding)
                face_quality_scores.append(quality_score)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing image {idx + 1}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to process image {idx + 1}: {str(e)}",
                )

        encodings_array = np.array(encodings_list)
        averaged_encoding = np.mean(encodings_array, axis=0)
        avg_quality = float(np.mean(face_quality_scores))

        logger.info(
            f"Successfully processed {len(files)} images. "
            f"Average quality score: {avg_quality:.2f}"
        )

        return {
            "success": True,
            "facialEncoding": averaged_encoding.tolist(),
            "metadata": {
                "images_processed": len(files),
                "average_quality": float(avg_quality),
                "quality_scores": [float(q) for q in face_quality_scores],
            },
        }

    async def _process_single_image(
        self, file: UploadFile, index: int
    ) -> Tuple[np.ndarray, float]:
        """Process a single image and return encoding with quality score"""

        # FILE TYPE VALIDATION
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for image {index + 1}: {file.content_type}",
            )

        image_data = await file.read()
        if not image_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Empty image file {index + 1}",
            )

        try:
            image = Image.open(io.BytesIO(image_data))
            image = self._preprocess_image(image)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not process image {index + 1}: {str(e)}",
            )

        # NUMPY ARRAY CONVERSION
        image_array = np.array(image)

        # DETECT FACES USING THE CNN MODEL(CAN BE CHANGE INTO HOG)
        face_locations = face_recognition.face_locations(image_array, model="cnn")

        if len(face_locations) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No face detected in image {index + 1}. "
                "Please ensure your face is clearly visible and well-lit.",
            )

        if len(face_locations) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Multiple faces detected in image {index + 1}. "
                "Please ensure only one face is visible.",
            )

        # VALIDATE FACE SIZE
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top

        if face_width < self.MIN_FACE_SIZE[0] or face_height < self.MIN_FACE_SIZE[1]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Face too small in image {index + 1}. "
                "Please move closer to the camera.",
            )

        # QUAILTY SCORE CALCULATION (0-100)
        quality_score = self._calculate_quality_score(image_array, face_locations[0])

        if quality_score < 40:
            logger.warning(
                f"Low quality score ({quality_score:.2f}) for image {index + 1}"
            )

        # ENCONDING EXTRACTION
        encodings = face_recognition.face_encodings(image_array, face_locations)

        if not encodings:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to extract facial features from image {index + 1}",
            )

        return encodings[0], quality_score

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for optimal face detection"""

        # RGB CONVERSION
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size

        if width < self.MIN_DIMENSION or height < self.MIN_DIMENSION:
            raise ValueError(
                f"Image too small ({width}x{height}). "
                f"Minimum size is {self.MIN_DIMENSION}x{self.MIN_DIMENSION}."
            )

        if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
            image.thumbnail(
                (self.MAX_DIMENSION, self.MAX_DIMENSION), Image.Resampling.LANCZOS
            )

        image = self._enhance_image(image)

        return image

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhancements to improve face detection"""

        # SHARPNESS
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        # CONTRAST
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)

        return image

    def _calculate_quality_score(
        self, image_array: np.ndarray, face_location: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate image quality score based on multiple factors.

        Returns: Score from 0-100
        """
        top, right, bottom, left = face_location
        face_region = image_array[top:bottom, left:right]

        # GRAYSCALE CONVERSION USED FOR ANALYSIS
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region

        gray_array = np.asarray(gray, dtype=np.float64)

        # CALCULATE SHARPNESS
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_array = np.asarray(laplacian, dtype=np.float64)
        laplacian_var = float(laplacian_array.var())
        sharpness_score = min(laplacian_var / 500 * 100, 100)

        # CALCULATE BRIGHTNESS
        brightness = float(gray_array.mean())
        brightness_score = 100 - abs(brightness - 127) / 127 * 100

        # CALCULATE CONTRST
        contrast = float(gray_array.std())
        contrast_score = min(contrast / 50 * 100, 100)

        face_area = (right - left) * (bottom - top)
        image_area = image_array.shape[0] * image_array.shape[1]
        size_ratio = face_area / image_area
        size_score = min(size_ratio / 0.15 * 100, 100)

        quality_score = (
            sharpness_score * 0.35
            + brightness_score * 0.25
            + contrast_score * 0.25
            + size_score * 0.15
        )

        return quality_score
