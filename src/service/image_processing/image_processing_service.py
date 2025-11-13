import base64
import binascii
import io
import logging
from typing import List

import face_recognition
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image

logging.basicConfig(level=logging.WARNING)


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
        try:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail=f"Invalid file type: {file.filename}"
                )

            image_data = await file.read()
            if not image_data:
                raise HTTPException(
                    status_code=400, detail=f"Empty or corrupted file: {file.filename}"
                )

            with Image.open(io.BytesIO(image_data)) as img:
                img = img.convert("RGB")

                img.thumbnail((640, 640), Image.Resampling.LANCZOS)

                image_array = np.asarray(img, dtype=np.uint8)

            del image_data

            face_locations = face_recognition.face_locations(image_array, model="hog")
            encodings = face_recognition.face_encodings(image_array, face_locations)

            if len(encodings) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected 1 face in {file.filename}, found {len(encodings)}",
                )

            return encodings[0]

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error processing {file.filename}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error processing {file.filename}: {e}"
            )
