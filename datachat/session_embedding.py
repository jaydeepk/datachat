from ast import Tuple
from importlib import metadata
from typing import Dict, List
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class SessionEmbedding:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    
    def generate(self, sessions: List[Dict]) -> List[tuple]:
        vector_embeddings: List[tuple]= []
        for session in sessions:
            text = f"Title: {session['title']}\nSpeaker: {session['nominator']}\nDate: {session['timeslot']}"
            embedding = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            ).data[0].embedding
            
            # Only use the fields we have in our data
            metadata = {
                'title': session['title'],
                'speaker': session['nominator'],
                'date': session['timeslot'],
                'type': session['type']
            }
            
            vector_id = f"session_{session['id']}"
            
            vector_embeddings.append((vector_id, embedding, metadata))
            
        return vector_embeddings

        
            
