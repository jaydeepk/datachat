from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os

from datachat.embeddings.models import model
from datachat.embeddings.models.model import Model
from datachat.store.vector_store import VectorStore

load_dotenv()

class DataChat:
   
    def __init__(
        self, 
        vector_store: VectorStore,
        model: Model,
        system_prompt: Optional[str] = None
    ):
        """Initialize DataChat with configurable parameters.
        
        Args:
            vector_store: Vector store for searching embeddings
            system_prompt: Custom system prompt for the chat model
            embedding_model: model to use for generating embeddings
        """
        self.vector_store = vector_store
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.model = model
    
    def search_sessions(self, query: str, top_k: int = 100) -> List[Dict]:
        """Search for relevant items using the query.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of relevant items with their metadata
        """
        query_embedding = self.model.create_embedding(query)
        return self.vector_store.search(query_embedding, top_k)
    
    def generate_response(self, user_query: str) -> str:
        """Generate a response based on the user query and relevant context.
        
        Args:
            user_query: User's question about the data
            
        Returns:
            Generated response from the chat model
        """
        context = self.search_sessions(user_query)
        return self.model.create_completion(context, user_query)