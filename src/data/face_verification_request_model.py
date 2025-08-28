from pydantic import BaseModel
from typing import List


class FaceVerificationRequest(BaseModel):
    uploaded_encoding: list[float]
    reference_encoding: list[float]
