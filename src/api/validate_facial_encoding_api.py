from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import logging

# models
from src.data.face_encoding_request_model import FaceEncodingRequest

# services
from src.service.validate_facial_encoding import FaceEncodingService

router = APIRouter()
service = FaceEncodingService()


@router.post("/validate-facial-encoding")
async def validate_facial_encoding(request: FaceEncodingRequest):
    """
    Validate and return facial encoding sent by the client.

    Returns:
        JSONResponse: Success response with validated face encoding or error details
    """
    try:
        encoding_list = service.validate_and_format_encoding(request.facialEncoding)

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
