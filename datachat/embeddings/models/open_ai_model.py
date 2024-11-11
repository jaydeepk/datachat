from datachat.embeddings.models.model import Model


from dotenv import load_dotenv
from openai import OpenAI


import os
from typing import List


class OpenAIModel(Model):
    """OpenAI's implementation of embedding model"""
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant.
        When displaying dates and times:
        - Always include both date and time if available in the format DD-MMM-YYYY HH:mm
        - Use 24-hour format for time
        
        For questions about total counts:
        - Return the actual count of items in the provided context
        - Be precise with numbers
        
        Ensure all relevant information from the context is included in your responses."""

    def __init__(self, system_prompt:str,  embedding_model_name: str, inference_model_name: str="gpt-4"):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.embedding_model_name = embedding_model_name
        self.inference_model_name = inference_model_name

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI's API"""
        response = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=text
        )
        return response.data[0].embedding
    
    
    def create_completion(self,  context:List[dict], user_query:str) -> str:
        """Generate completion for given query"""
        response = self.client.chat.completions.create(
            model= self.inference_model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Using the following context: {context}\n\nAnswer: {user_query}"}
            ]
        )
        return response.choices[0].message.content