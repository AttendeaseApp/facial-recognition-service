import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for validation errors.
    """
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
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
