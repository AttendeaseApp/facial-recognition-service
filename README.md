# Facial Recognition Microservice for RCIANS ATTENDEASE Backend Service

[![UNIT TESTS](https://github.com/AttendeaseApp/facial-recognition-service/actions/workflows/tests.yml/badge.svg)](https://github.com/AttendeaseApp/facial-recognition-service/actions/workflows/tests.yml)
[![Build and Push Docker Image](https://github.com/AttendeaseApp/facial-recognition-service/actions/workflows/docker-image.yml/badge.svg)](https://github.com/AttendeaseApp/facial-recognition-service/actions/workflows/docker-image.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)

A production-ready facial recognition microservice for **RCians Attendease**. This service provides robust endpoints for facial encoding extraction, validation, and comparison, extending the capabilities of the attendease-backend (Spring) service.

## Features

- **Face Encoding Extraction** - Extract 128-dimensional facial encodings from images
- **Face Verification** - Compare and authenticate facial encodings with configurable thresholds
- **Multi-Image Registration** - Register users with 5+ images for enhanced accuracy
- **Health Monitoring** - Built-in health checks and integrated test execution
- **Security Scans** - Automated dependency and code security scanning
- **Test Coverage** - Comprehensive test suite with 30+ unit tests
- **Docker Ready** - Containerized deployment support
- **Interactive API Docs** - Auto-generated Swagger and ReDoc documentation

## Prerequisites

- **Python** 3.10, 3.11, or 3.12 (3.13.6 recommended for development)
- **CMake** 3.16.0 or higher
- **pip** (Python package installer)

### System Dependencies (Windows)

- Install [CMake](https://cmake.org/download/)

## Setting up

### 1. Clone the Repository

```bash
git clone https://github.com/AttendeaseApp/facial-recognition-service.git
cd facial-recognition-service
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r dev-requirements.txt
```

### 4. Run the Application

```bash
uvicorn main:app --reload --port 8001
```

The application should be available at:
- **API**: http://127.0.0.1:8001
- **Swagger Documentation**: http://127.0.0.1:8001/docs
- **ReDoc Documentation**: http://127.0.0.1:8001/redoc

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/extract-face-encoding` | POST | Extract facial encoding from base64 image |
| `/verification/authenticate-face` | POST | Compare two facial encodings |
| `/extract-multiple-face-encodings` | POST | Register user using 5 images |

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health/status` | GET | Basic health check with dependencies |
| `/health/run-tests` | GET | Execute test suite and return results |
| `/health/test-summary` | GET | View cached test results |
| `/health/full` | GET | Comprehensive health check with tests |


## Unit Testing

### Run All Tests

```bash
pytest tests/
```

### Run Tests via Health Endpoint

```bash
# Execute tests through the API
curl http://127.0.0.1:8001/health/run-tests

# View cached results
curl http://127.0.0.1:8001/health/test-summary
```

## Development

### Project Structure

```
facial-recognition-service/
├── .github/
│   └── workflows/           # CI/CD pipelines
├── src/
│   ├── api/
│   │   ├── facial_registration/
│   │   ├── facial_verification/
│   │   └── health_check/
│   ├── service/
│   │   └── image_processing/ # Services
│   └── data/                # Request/response models
├── tests/                   # Unit tests
│   └── test_image_processing_service.py
├── main.py                  # Application entry point
├── dev-requirements.txt     # Project dependencies
├── pyproject.toml          # Project configuration
├── Dockerfile
└── README.md
```

## Authors

**Jake Viado**

- GitHub: [@jakeeviado](https://github.com/jakeeviado)

---

**for RCIANS ATTENDEASE**
