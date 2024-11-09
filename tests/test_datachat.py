import json
import pytest
from datachat.data_chat import DataChat
import os

from datachat.session_embedder import SessionEmbedder

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
index_name = os.getenv('DATACHAT_INDEX')
embedder = SessionEmbedder(index_name)
embedder.embed_sessions(sessions)
data_chat_bot =  DataChat(index_name)   

def test_query_speaker_sessions():
    """Test querying sessions by speaker"""
    query = "What sessions is James Smith presenting?"
    response = data_chat_bot.generate_response(query)
    
    assert "Future of AI" in response
    assert "22-Oct-2024 at 09:00" in response

def test_query_date_sessions():
    """Test querying sessions by date"""
    query = "What sessions are happening on October 22nd?"
    response = data_chat_bot.generate_response(query)
    
    assert "Future of AI" in response
    assert "Agile in Practice" in response

def test_query_session_type():
    """Test querying by session type"""
    query = "What keynote sessions are there?"
    response = data_chat_bot.generate_response(query)
    
    assert "Future of AI" in response
    assert "James Smith" in response
    assert "keynote" in response.lower()

def test_query_non_existent_speaker():
    """Test querying for a speaker that doesn't exist"""
    query = "What sessions is Alice Brown presenting?"
    response = data_chat_bot.generate_response(query)
    
    assert "no sessions" in response.lower() or "not presenting" in response.lower()


