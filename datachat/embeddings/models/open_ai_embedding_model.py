from datachat.embeddings.models.embedding_model import EmbeddingModel


from dotenv import load_dotenv
from openai import OpenAI


import os
from typing import List


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI's implementation of embedding model"""

    def __init__(self, model_name: str):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_name = model_name

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI's API"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding