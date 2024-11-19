from fastapi import APIRouter, HTTPException, Path
from typing import Dict, Any, List
import json

from pydantic import BaseModel

from datachat.core import registry
from datachat.core.data_chat import DataChat
from datachat.core.registry import DocumentRegistry
from .models import ChatQuery, ChatResponse, UploadResponse

router = APIRouter()

data_chat = DataChat()
registry = DocumentRegistry.get_instance()


class UploadPayload(BaseModel):
    data: List[Dict[str, Any]]
    document_type: str
    system_prompt: str


@router.post("/datasets/{dataset_name}/upload")
async def upload_data(
    dataset_name: str = Path(
        ..., description="The name of the dataset to upload data to"
    ),
    payload: UploadPayload = None,
) -> UploadResponse:
    """Upload JSON data for a specific dataset"""
    data = payload.data
    document_type = payload.document_type
    try:
        document_class = registry.get(document_type)
    except KeyError:
        raise HTTPException(400, f"Unknown document type: {document_type}")

    try:
        documents = [document_class(item) for item in data]
        data_chat.register(dataset_name, documents, payload.system_prompt)
        return UploadResponse(
            message=f"Dataset '{dataset_name}' uploaded and processed successfully"
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to process upload: {str(e)}")


@router.post("/datasets/{dataset_name}/chat", response_model=ChatResponse)
async def chat(dataset_name: str, query: ChatQuery) -> ChatResponse:
    """Generate a response to user message"""
    try:
        response = data_chat.generate_response(dataset_name, query.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate response: {str(e)}")
