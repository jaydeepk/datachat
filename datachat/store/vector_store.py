from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    """Base class for vector stores"""
    
    @abstractmethod
    def upsert(self, vectors: List[tuple]) -> None:
        """Upsert vectors to store"""
        pass
    
    @abstractmethod
    def search(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        pass