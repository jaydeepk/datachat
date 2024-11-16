from abc import ABC, abstractmethod
from typing import Any, Dict


class Document(ABC):
    """Base class for defining document data to be stored in a vector db"""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the document."""
        pass

    @property
    @abstractmethod
    def text(self) -> str:
        """Text representation for embedding and semantic search."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Additional metadata to store with the document."""
        pass
