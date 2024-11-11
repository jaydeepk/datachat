import json
from typing import List, Dict, Any
import pytest
from datachat.data_chat import DataChat
from datachat.embeddings.models.open_ai_embedding_model import OpenAIEmbeddingModel
from tests.session_embedding import SessionEmbedding
from datachat.store.pinecone_store import PineconeStore

class TestDataChat:
    """Tests for the DataChat functionality with session data"""

    @pytest.fixture(scope="class")
    def sample_sessions(self) -> List[Dict[str, Any]]:
        """Fixture providing sample session data"""
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
        return json.loads(DATA)

    @pytest.fixture(scope="class")
    def data_chat_bot(self, sample_sessions: List[Dict[str, Any]], pinecone_index: str) -> DataChat:
        """Fixture setting up DataChat with embedded sessions"""
        embedding_model = OpenAIEmbeddingModel("text-embedding-ada-002")
        session_embedding = SessionEmbedding(embedding_model)
        vectors = session_embedding.create(sample_sessions)

        vector_store = PineconeStore(pinecone_index)
        vector_store.upsert(vectors)
        
        return DataChat(vector_store)

    @pytest.fixture(scope="class")
    def pinecone_index(self) -> str:
        """Fixture providing Pinecone index name"""
        import os
        index_name = os.getenv('DATACHAT_INDEX')
        if not index_name:
            pytest.skip("DATACHAT_INDEX environment variable not set")
        return index_name

    @pytest.mark.parametrize("query,expected_phrases", [
        (
            "What sessions is James Smith presenting?",
            ["Future of AI", "22-Oct-2024 at 09:00"]
        ),
        (
            "What sessions are happening on October 22nd?",
            ["Future of AI", "Agile in Practice"]
        ),
        (
            "Which are the keynote sessions?",
            ["Future of AI", "James Smith", "keynote"]
        )
    ])
    def test_successful_queries(self, data_chat_bot: DataChat, query: str, expected_phrases: List[str]):
        """Test various successful query scenarios"""
        # Generate response
        response = data_chat_bot.generate_response(query)
        
        # Print query and response
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")
        
        # Assert expected phrases
        for phrase in expected_phrases:
            assert phrase in response, f"Expected phrase '{phrase}' not found in response"

    def test_query_non_existent_speaker(self, data_chat_bot: DataChat):
        """Test querying for a speaker that doesn't exist"""
        query = "What sessions is Alice Brown presenting?"
        response = data_chat_bot.generate_response(query)
        
        # Print query and response
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")
        
        assert any(phrase in response.lower() for phrase in ["no sessions", "not presenting"]), \
            "Response should indicate no sessions found"