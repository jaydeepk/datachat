import json
from typing import List, Dict, Any
from deepeval import assert_test, evaluate
import pytest
from datachat.core.data_chat import DataChat
from tests.session_document import SessionDocument
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


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
        data_chat = None  # Initialize outside try block

        try:
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
            data_chat.register_dataset(
                "conf-sessions-test", session_documents, system_prompt
            )

            yield data_chat

        finally:
            if data_chat:  # Only attempt cleanup if setup was successful
                try:
                    data_chat.delete_dataset("conf-sessions-test")
                    print("\nSuccessfully deleted conf-sessions-test dataset")
                except Exception as e:
                    print(f"\nFailed to delete dataset: {e}")

    @pytest.mark.parametrize(
        "query,expected_answer",
        [
            (
                "What sessions is James Smith presenting?",
                """ James Smith is presenting a keynote titled "The Future of AI" on 22-Oct-2024 at 09:00""",
            ),
            (
                "What sessions are happening on October 22nd?",
                """On October 22nd, 2024, there are 2 sessions happening:
                    1. "The Future of AI" is a keynote by James Smith at 09:00.
                    2. "Agile in Practice" is a session by John Doe at 11:00.""",
            ),
            (
                "Of these how many sessions are on Agile?",
                """There is 1 session on Agile titled "Agile in Practice" by John Doe on 22-Oct-2024 at 11:00""",
            ),
            (
                "Which are the keynote sessions?",
                """The keynote session is "The Future of AI" by James Smith, which will be held on 22-Oct-2024 at 09:00""",
            ),
            (
                "What sessions is Alice Brown presenting?",
                """Based on the provided context, there are no sessions being presented by Alice Brown""",
            ),
        ],
    )
    def test_qa(
        self,
        data_chat: DataChat,
        query: str,
        expected_answer: str,
    ):
        """Test Q&A responses using multiple evaluation metrics"""
        # Get actual response
        actual_answer = data_chat.generate_response("conf-sessions-test", query)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_answer,
            expected_output=expected_answer,
        )

        print(f"\nQuery: {query}")
        print(f"Expected: {expected_answer}")
        print(f"Actual: {actual_answer}\n")

        assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])
