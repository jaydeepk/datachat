from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

from datachat.core.config import Config, PineconeConfig
from datachat.core.exceptions import VectorStoreError
from datachat.core.models import EmbeddingModel, OpenAIEmbedding

from .vector_store import VectorStore


class PineconeStore(VectorStore):
    """Pinecone vector store implementation"""

    def __init__(self, config: PineconeConfig):
        self.pc = Pinecone(api_key=config.api_key)
        self.config = config

    def get_index(self, index_name):
        indexes = self.pc.list_indexes()
        if index_name not in indexes.names():
            return self.pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.config.region),
            )
        return self.pc.Index(index_name)

    def upsert(self, index_name: str, vectors: List[tuple]) -> None:
        """Upsert vectors to Pinecone"""
        try:
            index = self.get_index(index_name)
            index.upsert(vectors=vectors)
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert vectors: {str(e)}")

    def search(
        self, index_name: str, query_vector: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone"""
        try:
            index = self.pc.Index(index_name)
            results = index.query(
                vector=query_vector, top_k=top_k, include_metadata=True
            )
            return [match.metadata for match in results.matches]
        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")
