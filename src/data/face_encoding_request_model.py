from pydantic import BaseModel


class FaceEncodingRequest(BaseModel):
    facialEncoding: List[float]
