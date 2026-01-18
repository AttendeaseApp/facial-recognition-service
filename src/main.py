import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# api
from src.api.facial_registration.facial_registration_router import (
    router as register_facial_encodings,
)
from src.api.facial_verification.compare_face_encoding_router import (
    router as authenticate_face_router,
)
from src.api.facial_verification.extract_face_encoding_router import (
    router as authenticate_user_face,
)
from src.api.health_check.health_check_router import (
    router as system_health_check,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request size limits.
    Prevents memory issues from excessively large uploads.
    """

    MAX_REQUEST_SIZE = 100 * 1024 * 1024

    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length:
                content_length = int(content_length)
                if content_length > self.MAX_REQUEST_SIZE:
                    logger.warning(
                        f"Request rejected: size {content_length} bytes exceeds "
                        f"limit of {self.MAX_REQUEST_SIZE} bytes"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "success": False,
                            "error": "Request Entity Too Large",
                            "message": f"Request size ({content_length / 1024 / 1024:.2f} MB) "
                            f"exceeds maximum allowed size "
                            f"({self.MAX_REQUEST_SIZE / 1024 / 1024:.0f} MB)",
                            "max_size_mb": self.MAX_REQUEST_SIZE / 1024 / 1024,
                        },
                    )

        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown events.
    """
    # service startup
    logger.info("Starting Facial Recognition Service API...")
    logger.info("Checking dependencies...")

    try:
        logger.info("All core dependencies loaded successfully")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")

    logger.info("Configuration:")
    logger.info(
        f"  - Max request size: {RequestSizeLimitMiddleware.MAX_REQUEST_SIZE / 1024 / 1024:.0f} MB"
    )
    logger.info("Service ready to accept requests")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Health Check: http://localhost:8000/health/status")
    logger.info("Run Tests: http://localhost:8000/health/run-tests")

    yield

    # service shutdown
    logger.info("Shutting down Facial Recognition Service API...")
    logger.info("Shutdown complete")


app = FastAPI(
    title="Facial Recognition Service API",
    description="""
    ## Facial Recognition Microservice

    A production-ready facial recognition service providing:

    ### Features
    * **Face Encoding Extraction** - Extract 128-dimensional facial encodings from images
    * **Face Verification** - Compare and authenticate facial encodings
    * **Multi-Image Registration** - Register users with multiple images for accuracy
    * **Health Monitoring** - Built-in health checks and test execution

    ### Endpoints Overview
    * `/extract-face-encoding` - Extract encoding from base64 image
    * `/verification/authenticate-face` - Compare two face encodings
    * `/extract-multiple-face-encodings` - Register with 5+ images
    * `/health/status` - Service health and dependencies
    * `/health/run-tests` - Execute test suite
    * `/health/full` - Comprehensive health check

    ### Limits
    * Maximum request size: 100 MB
    * Recommended individual image size: < 5 MB

    ### Author
    Rogationist Computer Society

    ### Version
    1.0.0
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(RequestSizeLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(authenticate_face_router, tags=["Face Verification"])

app.include_router(authenticate_user_face, tags=["Face Verification/Extraction"])

app.include_router(register_facial_encodings, tags=["Multi-Image Registration"])

app.include_router(system_health_check, tags=["Health Check & Testing"])


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information and quick links.
    """
    return {
        "service": "Facial Recognition Service API",
        "version": "1.0.0",
        "status": "operational",
        "author": "Rogationist Computer Society",
        "endpoints": {
            "documentation": "/docs",
            "alternative_docs": "/redoc",
            "health_check": "/health/status",
            "run_tests": "/health/run-tests",
            "full_health": "/health/full",
        },
        "features": [
            "Face encoding extraction",
            "Face verification",
            "Multi-image registration",
            "Health monitoring",
            "Integrated testing",
        ],
        "limits": {
            "max_request_size_mb": 100,
            "recommended_image_size_mb": 5,
            "max_images_per_request": 5,
        },
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for validation errors.
    """
    logger.warning(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation Error",
            "details": exc.errors(),
            "body": exc.body,
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """
    Custom handler for internal server errors.
    """
    logger.error(f"Internal error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler for better error visibility.
    """
    logger.error(
        f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": type(exc).__name__,
            "message": str(exc)
            if not isinstance(exc, Exception)
            else "An unexpected error occurred",
            "path": request.url.path,
        },
    )


# RUN THE SERVICE HERE
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        timeout_keep_alive=120,
        limit_concurrency=1000,
    )
