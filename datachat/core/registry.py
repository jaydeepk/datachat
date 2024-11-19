from typing import Dict, Optional, Type
from .document import Document

class DocumentRegistry:
    """Registry for document implementations."""
    
    _instance: Optional['DocumentRegistry'] = None
    
    def __init__(self):
        if DocumentRegistry._instance is not None:
            raise RuntimeError("Use DocumentRegistry.get_instance() to access the registry")
        self.document_types: Dict[str, Type[Document]] = {}
    
    @classmethod
    def get_instance(cls) -> 'DocumentRegistry':
        """Get the singleton instance of DocumentRegistry"""
        if cls._instance is None:
            cls._instance = DocumentRegistry()
        return cls._instance
    
    def register(self, name: str, document_class: Type[Document]) -> None:
        """Register a document implementation."""
        if name in self.document_types:
            raise ValueError(f"Document type '{name}' is already registered")
            
        if not issubclass(document_class, Document):
            raise TypeError(
                f"Document class must inherit from Document, got {document_class}"
            )
            
        self.document_types[name] = document_class
    
    def get(self, name: str) -> Type[Document]:
        """Get a registered document implementation."""
        if name not in self.document_types:
            raise KeyError(f"Document type '{name}' is not registered")
        return self.document_types[name]
    
    def list_registered(self) -> list[str]:
        """List all registered document type names."""
        return list(self.document_types.keys())
    
    def clear(self) -> None:
        """Clear all registered document types."""
        self.document_types.clear()
