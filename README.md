# Facial Recognition API Service

This project is a facial recognition service of RCians Attendease. It is a service that provides endpoints for validating and comparing facial encodings. It extends the capabilities of the attendease-backend(spring backend service).

## Prerequisites

- Python 3.13.6
- CMake 3.16.0
- pip (python package installer)

## Setup Instructions

1. Clone this repository, then

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

   then

   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies (already on the dev-requirements.txt):

   ```bash
   pip install -r dev-requirements.txt
   ```

## Usage

Run the application:

```bash
uvicorn src.main:app --reload --port 8001
```

The application will be available at `http://127.0.0.1:8001`.

The documentaion will be available at `http://127.0.0.1:8001/docs` or `http://127.0.0.1:8001/redoc`.
