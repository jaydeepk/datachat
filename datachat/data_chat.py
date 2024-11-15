from typing import List, Optional

from datachat.document import Document
from datachat.store.pinecone_store import PineconeStore
from datachat.store.vector_store import VectorStore
from datachat.config import Config
from datachat.models import OpenAIInference

from langchain.memory import ConversationBufferWindowMemory


class DataChat:

    def __init__(
        self,
        documents: List[Document],
        system_prompt: Optional[str],
        config: Optional[Config] = None,
        memory_size: int = 3,  # Keep last 3 message pairs by default
    ):
        """Initialize DataChat with documents and system prompt.

        Args:
            documents: List of documents to process
            system_prompt: System prompt for the chat model
            config: Optional configuration for OpenAI and Pinecone
            chat_config: Optional chat-specific configuration
        """

        if not documents:
            raise ValueError("Documents list cannot be empty")
        if not system_prompt:
            raise ValueError("System prompt cannot be empty")

        self.config = config or Config.load()
        self.vector_store = self._initialize_pinecone(documents)
        self.inference_model = OpenAIInference(self.config.openai, system_prompt)
        # Initialize simple window memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_size, return_messages=True
        )

    def _initialize_pinecone(self, documents: List[Document]) -> VectorStore:
        if documents is None:
            raise Exception("Please provide a list of documents")
        print()
        print("DataChat is initializing Pinecone db")
        pinecone_store = PineconeStore(self.config)
        print("Pinecone db initialized")
        print()
        print("Upserting doucments in Pinecone db...")
        pinecone_store.upsert(documents)
        print("Finished upserting doucments in Pinecone db")
        return pinecone_store

    def generate_response(self, user_query: str, top_k: int = 100) -> str:
        """Generate a response based on the user query and relevant context.

        Args:
            user_query: User's question about the data

        Returns:
            Generated response from the chat model
        """
        # Get relevant context from vector store
        context = self.vector_store.search(user_query, top_k)

        # Get conversation history
        history = self.memory.load_memory_variables({}).get("history", [])

        # Create prompt with history and context
        history_text = "\n".join(
            [
                (
                    f"Human: {msg.content}"
                    if msg.type == "human"
                    else f"Assistant: {msg.content}"
                )
                for msg in history
            ]
        )

        return self.inference_model.generate_response(context, user_query, history_text)
