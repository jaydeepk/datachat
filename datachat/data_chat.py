
from typing import Dict, List
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class   DataChat:
    def __init__(self, index_name: str):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = pc.Index(index_name)
    
    def search_sessions(self, query: str, top_k: int = 100) -> List[Dict]:
        query_embedding = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        ).data[0].embedding
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata for match in results.matches]
    
    def generate_response(self, user_query: str) -> str:
        context = self.search_sessions(user_query)
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a conference assistant. 
                When displaying dates and times:
                - Always include both date and time if available in the format DD-MMM-YYYY HH:mm
                - Use 24-hour format for time
                
                For questions about total counts:
                - Return the actual count of all sessions in the provided context
                - Be precise with numbers
                
                Ensure all relevant information from the context is included in your responses."""},
                {"role": "user", "content": f"Using this conference data: {context}\n\nAnswer: {user_query}"}
            ]
        )
        return response.choices[0].message.content