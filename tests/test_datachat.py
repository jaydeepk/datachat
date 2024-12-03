import json
from typing import List, Dict, Any
import pytest
from datachat.core.data_chat import DataChat
from tests.session_document import SessionDocument


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
    def data_chat(self, session_data: List[Dict[str, Any]]) -> DataChat:
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
        data_chat = DataChat()
        data_chat.register("conf-sessions-test", session_documents, system_prompt)

        yield data_chat

        # This code runs after all tests are done
        try:
            data_chat.delete_dataset("conf-sessions-test")
            print("\nSuccessfully deleted conf-sessions-test dataset")
        except Exception as e:
            print(f"\nFailed to delete test dataset: {e}")

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
        self, data_chat: DataChat, query: str, expected_phrases: List[str]
    ):
        """Test various successful query scenarios"""
        # Generate response
        response = data_chat.generate_response("conf-sessions-test", query)

        # Print query and response
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")

        # Assert expected phrases
        for phrase in expected_phrases:
            assert (
                phrase in response
            ), f"Expected phrase '{phrase}' not found in response"

    def test_query_non_existent_speaker(self, data_chat: DataChat):
        """Test querying for a speaker that doesn't exist"""
        query = "What sessions is Alice Brown presenting?"
        response = data_chat.generate_response("conf-sessions-test", query)

        # Print query and response
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")

        assert any(
            phrase in response.lower()
            for phrase in ["no sessions", "not presenting", "not listed as a speaker"]
        ), f"Response should indicate no sessions found but got: {response}"
