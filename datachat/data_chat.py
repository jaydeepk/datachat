from typing import Dict, List, Optional
from openai import OpenAI

from datachat.models import EmbeddingModel, InferenceModel
from datachat.store.vector_store import VectorStore

class DataChat:
   
    def __init__(
        self, 
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        inference_model: InferenceModel
    ):
        """Initialize DataChat with configurable parameters.
        
        Args:
            vector_store: Vector store for searching embeddings
            system_prompt: Custom system prompt for the chat model
            model: model to use for generating embeddings and completions
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.inference_model = inference_model
    
    def search(self, query: str, top_k: int = 100) -> List[Dict]:
        """Search for relevant items using the query.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of relevant items with their metadata
        """
        query_embedding = self.embedding_model.create_embedding(query)
        return self.vector_store.search(query_embedding, top_k)
    
    def generate_response(self, user_query: str) -> str:
        """Generate a response based on the user query and relevant context.
        
        Args:
            user_query: User's question about the data
            
        Returns:
            Generated response from the chat model
        """
        context = self.search(user_query)
        return self.inference_model.generate_response(context, user_query)