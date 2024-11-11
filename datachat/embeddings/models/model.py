from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        pass
    
    
    @abstractmethod
    def create_completion(self, system_prompt:str, context:List[dict], user_query:str) -> str:
        """Generate embedding for given text"""
        pass