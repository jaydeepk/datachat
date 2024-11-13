

from abc import ABC, abstractmethod
from typing import Any, Dict


class Document(ABC):
    """Base class for defining document data to be stored in a vector db"""

    @abstractmethod
    def get_id(self) -> str:
        """Create identifier for the embedded item"""
        pass

    @abstractmethod
    def get_text(self) -> str:
        """Convert structured item into text representation"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Extract metadata from structured item"""
        pass
