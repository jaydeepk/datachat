import json
from openai import embeddings
import pytest
from datachat.data_chat import DataChat
import os

from datachat.embeddings.models.open_ai_embedding_model import OpenAIEmbeddingModel
from tests.session_embedding import SessionEmbedding
from datachat.store import vector_store
from datachat.store.pinecone_store import PineconeStore

# Sample data
DATA = """
[
    {
        "id": "1",
        "title": "The Future of AI",
        "nominator": "James Smith",
        "timeslot": "2024-10-22 09:00:00",
        "duration": 60,
        "type": "keynote"
    },
    {
        "id": "2",
        "title": "Agile in Practice",
        "nominator": "John Doe",
        "timeslot": "2024-10-22 11:00:00",
        "duration": 45,
        "type": "session"
    },
    {
        "id": "3",
        "title": "DevOps Transformation",
        "nominator": "Jane Smith",
        "timeslot": "2024-10-23 10:00:00",
        "duration": 45,
        "type": "session"
    }
]
"""


sessions = json.loads(DATA)
embedding_model = OpenAIEmbeddingModel("text-embedding-ada-002")
session_embedding = SessionEmbedding(embedding_model)
vectors = session_embedding.create(sessions)

vector_store = PineconeStore(os.getenv('DATACHAT_INDEX'))
vector_store.upsert(vectors)

data_chat_bot =  DataChat(vector_store)   

def test_query_speaker_sessions():
    """Test querying sessions by speaker"""
    query = "What sessions is James Smith presenting?"
    response = data_chat_bot.generate_response(query)
    print_question_and_answer(query, response)
    
    assert "Future of AI" in response
    assert "22-Oct-2024 at 09:00" in response

def test_query_date_sessions():
    """Test querying sessions by date"""
    query = "What sessions are happening on October 22nd?"
    response = data_chat_bot.generate_response(query)
    print_question_and_answer(query, response)
    
    assert "Future of AI" in response
    assert "Agile in Practice" in response

def test_query_session_type():
    """Test querying by session type"""
    query = "Which are the keynote sessions?"
    response = data_chat_bot.generate_response(query)
    print_question_and_answer(query, response)
    
    assert "Future of AI" in response
    assert "James Smith" in response
    assert "keynote" in response.lower()

def test_query_non_existent_speaker():
    """Test querying for a speaker that doesn't exist"""
    query = "What sessions is Alice Brown presenting?"
    response = data_chat_bot.generate_response(query)
    print_question_and_answer(query, response)
    
    assert "no sessions" in response.lower() or "not presenting" in response.lower()
    
def print_question_and_answer(question:str, answer:str):
    print()
    print(question)
    print(answer)
    print()


