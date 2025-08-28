from pydantic import BaseModel
from typing import List


class FaceEncodingRequest(BaseModel):
    facialEncoding: list[float]
