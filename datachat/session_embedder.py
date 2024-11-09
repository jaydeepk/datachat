from typing import Dict, List
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class SessionEmbedder:
    def __init__(self, index_name: str):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = index_name
        self.ensure_index_exists()
        self.index = self.pc.Index(index_name)
    
    def ensure_index_exists(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
    
    def embed_sessions(self, sessions: List[Dict]):
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
                'date': session['timeslot']
            }
            
            vector_id = f"session_{session['id']}"
        
            self.index.upsert(vectors=[(
                vector_id,   
                embedding,   
                metadata    
            )])