from abc import ABC, abstractmethod
from typing import List, Dict, Any

from datachat.core.document import Document


class VectorStore(ABC):
    """Base class for vector stores"""

    @abstractmethod
    def upsert(self, index_name: str, vectors: List[tuple]) -> None:
        """Upsert vectors to store"""
        pass

    @abstractmethod
    def search(self, index_name: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        pass
