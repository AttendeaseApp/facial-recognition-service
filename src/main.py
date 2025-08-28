from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import logging
import face_recognition
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)

from data.face_encoding_request_model import FaceEncodingRequest
from data.face_verification_request_model import FaceVerificationRequest

app = FastAPI(
    title="Facial Recognition Service API",
    description="Facial recognition service api for encoding validation and face encoding comparison",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.post("/v1/validate-facial-encoding")
async def facial_registration(request: FaceEncodingRequest):
    """
    Validate and return facial encoding sent by the client.

    Args:
        request (FaceEncodingRequest): JSON body containing faceEncoding (list of floats)

    Returns:
        JSONResponse: Success response with validated face encoding or error details
    """
    try:
        face_encoding_input = request.facialEncoding

        if isinstance(face_encoding_input, np.ndarray):
            face_encoding_input = face_encoding_input.tolist()
        elif not isinstance(face_encoding_input, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input faceEncoding must be a list of floats.",
            )
        faceEncoding = np.array(face_encoding_input, dtype=np.float64)
        if faceEncoding is None or len(faceEncoding) != 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid face encoding. Expected a 128-dimensional vector.",
            )
        if not np.all(np.isfinite(faceEncoding)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Face encoding contains invalid values (e.g., NaN or infinite).",
            )
        encoding_list = faceEncoding.tolist()
        logging.info("Face encoding validated successfully for registration.")
        return {
            "success": True,
            "message": "Face encoding validated",
            "facialEncoding": encoding_list,
        }
    except HTTPException as e:
        logging.error(
            f"HTTPException during facial registration: {e.detail}", exc_info=True
        )
        return JSONResponse(
            status_code=e.status_code, content={"success": False, "error": e.detail}
        )
    except Exception as e:
        logging.error(
            f"Unexpected error during facial registration: {e}", exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": f"An unexpected server error occurred: {str(e)}",
            },
        )


@app.post("/v1/authenticate-face")
async def compare_face_encoding(request: FaceVerificationRequest):
    """
    Compare two facial encodings to verify if they match.

    Args:
        request (FaceVerificationRequest): JSON body containing uploaded_encoding and reference_encoding

    Returns:
        JSONResponse: Verification result with confidence score or error details
    """
    try:
        uploaded_encoding = np.array(request.uploaded_encoding, dtype=np.float64)
        reference_encoding = np.array(request.reference_encoding, dtype=np.float64)

        if uploaded_encoding is None or reference_encoding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or missing encoding data",
            )
        if len(uploaded_encoding) != 128 or len(reference_encoding) != 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid encoding dimensions. Expected 128-dimensional vectors.",
            )
        if not np.all(np.isfinite(uploaded_encoding)) or not np.all(
            np.isfinite(reference_encoding)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Encodings contain invalid values (e.g., NaN or infinite).",
            )
        face_distance = float(np.linalg.norm(uploaded_encoding - reference_encoding))
        is_match = face_distance < 0.6
        confidence = max(0.0, 1.0 - face_distance)
        logging.info(
            f"Face verification completed. Match: {is_match}, Confidence: {confidence}"
        )
        return {
            "success": True,
            "verified": True,
            "is_face_matched": bool(is_match),
            "confidence": float(confidence),
            "face_distance": float(face_distance),
        }
    except HTTPException as e:
        logging.error(
            f"HTTPException during face verification: {e.detail}", exc_info=True
        )
        return JSONResponse(
            status_code=e.status_code, content={"success": False, "error": e.detail}
        )
    except Exception as e:
        logging.error(f"Unexpected error during face verification: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": f"An unexpected server error occurred: {str(e)}",
            },
        )


# used for testing only
@app.post("/v1/extract-face-encoding")
async def extract_face_encoding(file: UploadFile = File(...)):
    """
    Extract face encoding from an uploaded image.

    Args:
        file (UploadFile): Image file uploaded via multipart form data

    Returns:
        JSONResponse: Success response with extracted face encoding or error details
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or missing file type. Only image files are supported.",
            )

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        encodings = face_recognition.face_encodings(image_array)

        if not encodings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No faces detected in the image.",
            )
        if len(encodings) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multiple faces detected. Please upload an image with exactly one face.",
            )
        face_encoding = encodings[0]
        if len(face_encoding) != 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid face encoding. Expected a 128-dimensional vector.",
            )
        if not np.all(np.isfinite(face_encoding)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Face encoding contains invalid values (e.g., NaN or infinite).",
            )
        encoding_list = [str(x) for x in face_encoding.tolist()]
        logging.info("Face encoding extracted successfully from image.")
        return {
            "success": True,
            "message": "Face encoding extracted",
            "facialEncoding": encoding_list,
        }
    except HTTPException as e:
        logging.error(
            f"HTTPException during face encoding extraction: {e.detail}", exc_info=True
        )
        return JSONResponse(
            status_code=e.status_code, content={"success": False, "error": e.detail}
        )
    except Exception as e:
        logging.error(
            f"Unexpected error during face encoding extraction: {e}", exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": f"An unexpected server error occurred: {str(e)}",
            },
        )
