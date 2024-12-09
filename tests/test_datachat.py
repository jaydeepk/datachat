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
                "type": "keynote",
                "abstract": "Join us for a comprehensive exploration of artificial intelligence's trajectory over the next decade. This keynote session will delve deep into the transformative potential of AI and its implications for technology, business, and society. We'll begin by examining the current state of AI technology, particularly the revolutionary impact of large language models and their evolution towards more sophisticated multimodal capabilities. The discussion will cover recent breakthroughs in neural architectures, training methodologies, and the emerging paradigm of few-shot learning. We'll analyze how these advances are pushing the boundaries of what's possible in natural language processing, computer vision, and decision-making systems. Special attention will be given to the development of more efficient and environmentally sustainable AI models, including techniques for model compression and energy-efficient training. The session will explore the growing importance of AI governance and ethics, discussing frameworks for responsible AI development and deployment. We'll examine real-world case studies of AI implementation across various industries, from healthcare and finance to manufacturing and creative industries. Particular emphasis will be placed on the challenges and opportunities in making AI more accessible and democratized, including the role of AI-as-a-service platforms and the importance of reducing barriers to entry for smaller organizations. The talk will also address critical concerns about AI safety, bias mitigation, and the need for transparent and explainable AI systems. We'll discuss emerging approaches to building more robust and trustworthy AI systems, including advances in interpretable machine learning and techniques for ensuring AI alignment with human values. Looking ahead, we'll explore emerging trends such as hybrid AI systems that combine symbolic and neural approaches, the potential impact of quantum computing on AI capabilities, and the evolution of human-AI collaboration. The session will conclude with practical insights on how organizations can prepare for and adapt to the rapidly evolving AI landscape, including strategies for talent development, infrastructure planning, and ethical considerations in AI adoption.",
                "learning_outcome": "• Understand key trends shaping the future of AI and their potential impact on various industries• Identify practical applications of advanced AI technologies for your organization• Learn strategies for responsible AI adoption and governance• Gain insights into AI safety, ethics, and bias mitigation• Develop a framework for preparing your organization for AI transformation• Master the concepts of AI democratization and accessibility"
            },
            {
                "id": "2",
                "title": "Agile in Practice",
                "nominator": "John Doe",
                "timeslot": "2024-10-22 11:00:00",
                "duration": 45,
                "type": "session",
                "abstract": "This comprehensive session delves into the practical implementation of Agile methodologies in modern software development environments. We'll begin by examining the fundamental shifts in mindset required for successful Agile adoption, moving beyond theoretical frameworks to real-world applications. The discussion will cover extensive case studies from organizations of varying sizes, from startups to enterprise-level companies, highlighting both successes and failures in Agile transformation. We'll analyze the critical factors that determine the success of Agile implementations, including team dynamics, organizational culture, and leadership support. Special attention will be given to common challenges teams face during the transition to Agile, such as resistance to change, maintaining consistency across distributed teams, and balancing Agile principles with organizational constraints. The session will provide detailed insights into effective sprint planning techniques, including capacity planning, story point estimation, and backlog refinement strategies. We'll explore advanced topics in sprint execution, such as managing dependencies, handling technical debt, and maintaining sustainable development practices. Particular emphasis will be placed on the role of effective communication in Agile teams, including techniques for remote collaboration, cross-functional team coordination, and stakeholder management. The presentation will cover various Agile frameworks beyond Scrum, including Kanban, Scrumban, and SAFe, discussing when and how to adapt these methodologies to different contexts. We'll examine tools and technologies that support Agile practices, from project management software to automation tools, and discuss their integration into existing workflows. The session will also address the crucial aspect of measuring Agile success, introducing various metrics and KPIs that teams can use to track their progress and identify areas for improvement. We'll discuss advanced retrospective techniques that go beyond basic formats, helping teams to continuously evolve and adapt their practices. The presentation will include strategies for scaling Agile practices across multiple teams and departments, including coordination mechanisms, shared practices, and governance structures. Real-world examples will demonstrate how organizations have successfully implemented these scaling strategies while maintaining agility and efficiency.",
                "learning_outcome": "• Master practical techniques for implementing Agile methodologies in real-world scenarios• Learn effective sprint planning and execution strategies• Develop skills in story point estimation and capacity planning• Understand how to measure and improve team velocity• Gain expertise in conducting effective retrospectives• Master techniques for scaling Agile across multiple teams• Learn strategies for managing distributed Agile teams"
            },
            {
                "id": "3",
                "title": "DevOps Transformation",
                "nominator": "Jane Smith",
                "timeslot": "2024-10-23 10:00:00",
                "duration": 45,
                "type": "session",
                "abstract": "This in-depth session provides a comprehensive guide to successfully implementing DevOps practices in modern organizations. We'll begin by exploring the fundamental principles of DevOps, moving beyond the buzzwords to understand the cultural and technical foundations that drive successful transformations. The presentation will cover extensive real-world examples of DevOps implementations, examining both successful transformations and common pitfalls to avoid. We'll dive deep into the technical aspects of DevOps, including detailed discussions of continuous integration and continuous deployment (CI/CD) pipelines, automated testing strategies, and infrastructure as code (IaC) practices. The session will explore various toolchains and technologies that support DevOps practices, from version control systems and build tools to container orchestration platforms and monitoring solutions. Special attention will be given to security integration in the DevOps pipeline, discussing strategies for implementing DevSecOps and ensuring security is built into every stage of the software delivery process. We'll examine advanced topics such as microservices architecture, containerization, and cloud-native development, discussing how these technologies enable and enhance DevOps practices. The presentation will cover strategies for monitoring and observability in complex distributed systems, including logging, metrics collection, and distributed tracing. We'll discuss approaches to incident management and post-mortem analysis, helping teams learn from failures and build more resilient systems. The session will address the human aspects of DevOps transformation, including team structure, roles and responsibilities, and strategies for breaking down silos between development and operations teams. We'll explore change management techniques that help organizations navigate the cultural shifts required for successful DevOps adoption. The presentation will include detailed case studies of organizations that have successfully implemented DevOps at scale, examining their journey, challenges faced, and lessons learned. We'll discuss strategies for measuring DevOps success, including key metrics and KPIs that help teams track their progress and identify areas for improvement. The session will also cover advanced topics such as chaos engineering, site reliability engineering (SRE) practices, and the integration of artificial intelligence and machine learning in DevOps workflows.",
                "learning_outcome": "• Understand the key principles and practices of successful DevOps transformation• Master the implementation of CI/CD pipelines and automated testing• Learn effective strategies for implementing Infrastructure as Code• Develop expertise in containerization and microservices architecture• Gain skills in DevOps toolchain selection and integration• Understand how to measure and improve DevOps metrics• Master techniques for fostering a DevOps culture• Learn strategies for implementing DevSecOps practices"
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
                    Base rules:
                    - Answer ONLY what was asked
                    - For session queries, include ONLY title, speaker, date and time
                    - For learning outcome queries, return outcomes ONLY for the specific session being discussed
                    - Use DD-MMM-YYYY HH:mm format for dates/times
                    - Use 24-hour format for time

                    For follow-up questions:
                    - Consider ONLY the session from the previous response
                    - Ignore other sessions even if they match the query

                    For count queries:
                    - Return exact numbers from the provided context"""

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
                "What are its learning outcomes?",
                """ The learning outcomes are: Understand key trends shaping the future of AI and their potential impact on various industries• Identify practical applications of advanced AI technologies for your organization• Learn strategies for responsible AI adoption and governance• Gain insights into AI safety, ethics, and bias mitigation• Develop a framework for preparing your organization for AI transformation• Master the concepts of AI democratization and accessibility""",
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
