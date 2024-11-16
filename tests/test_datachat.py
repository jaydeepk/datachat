import json
from typing import List, Dict, Any
from xml.dom.minidom import Document
import pytest
from datachat.config import Config
from datachat.data_chat import DataChat
from datachat.models import OpenAIInference
from datachat.models import OpenAIEmbedding
from tests.session_document import SessionDocument
from datachat.store.pinecone_store import PineconeStore


class TestDataChat:
    """Tests for the DataChat functionality with session data"""

    @pytest.fixture(scope="class")
    def session_data(self) -> List[Dict[str, Any]]:
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
    def session_data_chat(self, session_data: List[Dict[str, Any]]) -> DataChat:
        """Fixture setting up DataChat with embedded sessions"""
        system_prompt = """You are a conference assistant. 
                When displaying dates and times:
                - Always include both date and time if available in the format DD-MMM-YYYY HH:mm
                - Use 24-hour format for time
                
                For questions about total counts:
                - Return the actual count of all sessions in the provided context
                - Be precise with numbers
                
                Ensure all relevant information from the context is included in your responses."""

        session_documents = [SessionDocument(session) for session in session_data]
        return DataChat(documents=session_documents, system_prompt=system_prompt)

    @pytest.fixture(scope="class")
    def pinecone_index(self) -> str:
        """Fixture providing Pinecone index name"""
        import os

        index_name = os.getenv("DATACHAT_INDEX")
        if not index_name:
            pytest.skip("DATACHAT_INDEX environment variable not set")
        return index_name

    @pytest.mark.parametrize(
        "query,expected_phrases",
        [
            (
                "What sessions is James Smith presenting?",
                ["Future of AI", "22-Oct-2024", "09:00"],
            ),
            (
                "What sessions are happening on October 22nd?",
                ["Future of AI", "Agile in Practice"],
            ),
            (
                "Of these how many sessions are on Agile?",
                ["Agile in Practice"],
            ),
            (
                "Which are the keynote sessions?",
                ["Future of AI", "James Smith", "keynote"],
            ),
        ],
    )
    def test_successful_queries(
        self, session_data_chat: DataChat, query: str, expected_phrases: List[str]
    ):
        """Test various successful query scenarios"""
        # Generate response
        response = session_data_chat.generate_response(query)

        # Print query and response
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")

        # Assert expected phrases
        for phrase in expected_phrases:
            assert (
                phrase in response
            ), f"Expected phrase '{phrase}' not found in response"

    def test_query_non_existent_speaker(self, session_data_chat: DataChat):
        """Test querying for a speaker that doesn't exist"""
        query = "What sessions is Alice Brown presenting?"
        response = session_data_chat.generate_response(query)

        # Print query and response
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")

        assert any(
            phrase in response.lower()
            for phrase in ["no sessions", "not presenting", "not listed as a speaker"]
        ), "Response should indicate no sessions found"
