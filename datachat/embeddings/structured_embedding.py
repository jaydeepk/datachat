from datachat.embeddings.models.model import Model


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class StructuredEmbedding(ABC):
    """Base class for converting structured data into semantic embeddings"""

    def __init__(self, model: Model):
        self.model = model

    @abstractmethod
    def get_text(self, item: Dict[str, Any]) -> str:
        """Convert structured item into text representation"""
        pass

    @abstractmethod
    def get_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from structured item"""
        pass

    @abstractmethod
    def get_id(self, item: Dict[str, Any]) -> str:
        """Create identifier for the embedded item"""
        pass

    
    def create(self, items: List[Dict[str, Any]]) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """Convert structured items into semantic embeddings with metadata"""
        return [
            (
                self.get_id(item),
                self.model.create_embedding(self.get_text(item)),
                self.get_metadata(item)
            )
            for item in items
        ]