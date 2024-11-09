import os
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from datachat.exceptions import VectorStoreError

from .vector_store import VectorStore

class PineconeStore(VectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self, index_name: str):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = index_name
        self.ensure_index_exists()
        self.index = self.pc.Index(index_name)
    
        
    def ensure_index_exists(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.getenv('PINECONE_ENVIRONMENT')
                )
            )
    
    def upsert(self, vectors: List[tuple]) -> None:
        """Upsert vectors to Pinecone"""
        try:
            self.index.upsert(vectors=vectors)
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert vectors: {str(e)}")
        
    
    def search(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone"""
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )
            return [match.metadata for match in results.matches]
        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")
    