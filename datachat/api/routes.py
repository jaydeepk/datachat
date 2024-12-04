from fastapi import APIRouter, Depends, HTTPException, Path
from typing import Dict, Any, List, Optional
import json

from pydantic import BaseModel

from datachat.core import registry
from datachat.core.data_chat import DataChat
from datachat.core.registry import DocumentRegistry
from .models import ChatQuery, ChatResponse, UploadResponse
import traceback


class DataChatManager:
    _instance: Optional[DataChat] = None

    @classmethod
    def get_instance(cls) -> DataChat:
        if cls._instance is None:
            cls._instance = DataChat()
        return cls._instance


class UploadPayload(BaseModel):
    data: List[Dict[str, Any]]
    document_type: str
    system_prompt: str


router = APIRouter()
registry = DocumentRegistry.get_instance()
chat_manager = DataChatManager()


async def get_data_chat() -> DataChat:
    return chat_manager.get_instance()


@router.post("/datasets/{dataset_name}/upload")
async def upload_data(
    data_chat: DataChat = Depends(get_data_chat),
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
        data_chat.register_dataset(dataset_name, documents, payload.system_prompt)
        return UploadResponse(
            message=f"Dataset '{dataset_name}' uploaded and processed successfully"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process upload: {str(e)}")


@router.post("/datasets/{dataset_name}/chat", response_model=ChatResponse)
async def chat(
    dataset_name: str, query: ChatQuery, data_chat: DataChat = Depends(get_data_chat)
) -> ChatResponse:
    """Generate a response to user message"""
    try:
        response = data_chat.generate_response(dataset_name, query.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate response: {str(e)}")
