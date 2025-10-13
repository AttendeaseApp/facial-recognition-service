import numpy as np
from typing import List
import logging
from fastapi import HTTPException, status, UploadFile
from PIL import Image
import io
import base64
import binascii
import face_recognition

logging.basicConfig(level=logging.INFO)


class ImageProcessingService:
    """
    Service layer for extracting a single facial encoding from an image (Base64).
    """

    def extract_encoding_from_base64(
        self,
        image_base64: str,
    ) -> dict:
        try:
            # base64 decoding
            header_removed = (
                image_base64.split(",")[1] if "," in image_base64 else image_base64
            )
            image_data = base64.b64decode(header_removed)

            if not image_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty or invalid image data after decoding.",
                )

            # image Processing
            try:
                image = Image.open(io.BytesIO(image_data))
                image = image.convert("RGB")
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not process image file.",
                )

            image_array = np.array(image)

            # encoding extraction
            encodings = face_recognition.face_encodings(image_array)

            if not encodings:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No faces detected in the image.",
                )
            if len(encodings) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Multiple faces detected.",
                )

            encoding = encodings[0]
            encoding_list: List[float] = encoding.tolist()

            return {"success": True, "facialEncoding": encoding_list}

        except (binascii.Error, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Base64 format: {str(e)}",
            )
        except HTTPException:
            raise
        except Exception as e:
            logging.error(
                f"Unexpected error during encoding extraction: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected server error occurred: {str(e)}",
            )

    async def extract_single_encoding(self, file: UploadFile) -> np.ndarray:
        """
        Extract face encoding from a single uploaded image file (multipart form data).

        Returns: np.ndarray (128-dimensional vector)
        Raises: HTTPException on file/face errors.
        """
        try:
            # 1. Validation and Read
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type for {file.filename}.",
                )
            image_data = await file.read()
            if not image_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Empty or corrupted image file: {file.filename}.",
                )

            # 2. Process image with PIL
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process image {file.filename}. File might be corrupted.",
                )

            # 3. Extract encoding with face_recognition
            image_array = np.array(image)
            encodings = face_recognition.face_encodings(image_array)

            if len(encodings) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No face detected in image: {file.filename}.",
                )
            if len(encodings) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Multiple faces detected in image: {file.filename}.",
                )

            face_encoding = encodings[0]

            # 4. Final encoding structure check
            if len(face_encoding) != 128 or not np.all(np.isfinite(face_encoding)):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Extracted face encoding for {file.filename} is invalid.",
                )

            logging.info(f"Successfully extracted face encoding from {file.filename}")
            return face_encoding

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred while processing {file.filename}.",
            )
