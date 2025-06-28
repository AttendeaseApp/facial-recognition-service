from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
import cv2
import face_recognition
import numpy as np

from src.services.compare_encoding_service import compare_student_face_encoding

app = FastAPI()

@app.post("/v1/get-face-encoding")
async def verify_attendance(
    image: UploadFile = File(...),
):
    try:
        image_data = await image.read()
        image_array = cv2.imdecode(
            np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
        )

        if image_array is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid image file"},
            )

        face_locations = face_recognition.face_locations(image_array)

        if len(face_locations) != 1:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid face detection"},
            )

        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        if not face_encodings:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No face encoding found"},
            )

        print("Face encoding:", face_encodings[0])

        return {"success": True, "faceEncoding": face_encodings[0].tolist()}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@app.get("/v1/compare-face-encoding")
async def compare_face_encoding(
    face_encoding: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        face_encoding_list = list(map(float, face_encoding.split(",")))
        known_encoding = np.load("test_encoding.npy")

        unknown_image = face_recognition.load_image_file(image.file)
        encoding = face_recognition.face_encodings(image)[0]
        np.save("test_encoding.npy", encoding)

        result = compare_student_face_encoding(known_encoding, unknown_image)
        return {"success": True, "match": True, "result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )
