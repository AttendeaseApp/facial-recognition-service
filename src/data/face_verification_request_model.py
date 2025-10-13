from pydantic import BaseModel
from typing import List


class FaceVerificationRequest(BaseModel):
    uploaded_encoding: List[float]
    reference_encoding: List[float]
