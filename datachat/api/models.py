from pydantic import BaseModel


class ChatQuery(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class UploadResponse(BaseModel):
    message: str
