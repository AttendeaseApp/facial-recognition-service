import time
from datetime import datetime

from fastapi import APIRouter, status

router = APIRouter()
startup_time = time.time()


@router.get("/health/status", tags=["Health"], status_code=status.HTTP_200_OK)
async def detailed_status():
    """
    Detailed health status endpoint.
    Provides comprehensive information about the service health.
    """
    uptime_seconds = time.time() - startup_time

    return {
        "status": "healthy",
        "service": "Facial Recognition Microservice",
        "author": "Rogationist Computer Society",
        "version": "1.0.0",
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "face_verification": "should_be_operational",
            "face_extraction": "should_be_operational",
            "facial_registration": "should_be_operational",
        },
    }
