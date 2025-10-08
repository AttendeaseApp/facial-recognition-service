from pydantic import BaseModel


class FaceEncodingRequest(BaseModel):
    facialEncoding: list[float]
