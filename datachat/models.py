from abc import ABC, abstractmethod
from curses.ascii import EM
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from datachat.config import OpenAIConfig


class EmbeddingModel(ABC):
    """Model for generating embeddings"""
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        pass
    

class InferenceModel(ABC):
    """Model for performing inference/generation tasks"""
    @abstractmethod
    def generate_response(
        self, 
        context: List[dict], 
        user_query: str
    ) -> str:
        """Generate a response based on context and query"""
        pass
    
    
class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, config:OpenAIConfig, model_name: str = "text-embedding-ada-002"):
        load_dotenv()
        self.client = OpenAI(api_key=config.api_key)
        self.model_name = model_name
    
    def create_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    
class OpenAIInference(InferenceModel):
    """OpenAI's implementation of embedding model"""
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant.
        When displaying dates and times:
        - Always include both date and time if available in the format DD-MMM-YYYY HH:mm
        - Use 24-hour format for time
        
        For questions about total counts:
        - Return the actual count of items in the provided context
        - Be precise with numbers
        
        Ensure all relevant information from the context is included in your responses."""

    def __init__(self, config:OpenAIConfig, system_prompt:str,  model_name: str="gpt-4"):
        load_dotenv()
        self.client = OpenAI(api_key=config.api_key)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.model_name = model_name

        
    def generate_response(self,  context:List[dict], user_query:str) -> str:
        """Generate completion for given query"""
        response = self.client.chat.completions.create(
            model= self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Using the following context: {context}\n\nAnswer: {user_query}"}
            ]
        )
        return response.choices[0].message.content