from logging import config
import os
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pydantic import conbytes

from datachat.config import Config, PineconeConfig
from datachat.exceptions import VectorStoreError
from datachat.models import EmbeddingModel, OpenAIEmbedding
from datachat.document import Document

from .vector_store import VectorStore


class PineconeStore(VectorStore):
    """Pinecone vector store implementation"""

    def __init__(self, config: Config, embedding_model: EmbeddingModel = None):
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else OpenAIEmbedding(config.openai)
        )
        self.pc = Pinecone(api_key=config.pinecone.api_key)
        self.config = config
        self.ensure_index_exists()
        self.index = self.pc.Index(config.pinecone.index_name)

    def ensure_index_exists(self):
        if self.config.pinecone.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.config.pinecone.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.config.pinecone.region),
            )

    def upsert(self, documents: List[Document]) -> None:
        """Upsert vectors to Pinecone"""
        try:
            vectors = [
                (
                    document.get_id(),
                    self.embedding_model.create_embedding(document.get_text()),
                    document.get_metadata(),
                )
                for document in documents
            ]
            self.index.upsert(vectors=vectors)
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert vectors: {str(e)}")

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone"""
        try:
            query_vector = self.embedding_model.create_embedding(query)
            results = self.index.query(
                vector=query_vector, top_k=top_k, include_metadata=True
            )
            return [match.metadata for match in results.matches]
        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")
