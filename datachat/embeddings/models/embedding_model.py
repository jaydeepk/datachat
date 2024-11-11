from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        pass