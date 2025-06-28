# Facial Recognition Service

This project is a facial recognition service built using FastAPI. It provides an endpoints to verify attendance by processing image uploads and comparing face encodings.

## Setup Instructions

1. Clone the repository:

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```bash
uvicorn src.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.
