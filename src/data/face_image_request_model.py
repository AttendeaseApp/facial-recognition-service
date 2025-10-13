from pydantic import BaseModel


class FaceImageRequest(BaseModel):
    image_base64: str
