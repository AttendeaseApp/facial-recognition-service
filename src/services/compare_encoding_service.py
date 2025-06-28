import numpy as np
import face_recognition


def get_student_face_encoding(student_id: str):
    raise NotImplementedError("get_student_face_encoding not implemented")


def compare_student_face_encoding(
    known_encoding: np.ndarray, unknown_image: np.ndarray, tolerance: float = 0.6
) -> bool:

    unknown_encodings = face_recognition.face_encodings(unknown_image)
    if not unknown_encodings:
        return False

    results = face_recognition.compare_faces(
        [known_encoding], unknown_encodings[0], tolerance=tolerance
    )
    return results[0]
