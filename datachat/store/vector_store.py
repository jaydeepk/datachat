from abc import ABC, abstractmethod
from typing import List, Dict, Any

from datachat.document import Document

class VectorStore(ABC):
    """Base class for vector stores"""
    
    @abstractmethod
    def upsert(self, documents: List[Document]) -> None:
        """Upsert vectors to store"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        pass