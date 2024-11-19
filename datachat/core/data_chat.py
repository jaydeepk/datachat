from asyncio import sleep
from typing import List, Optional

from attr import dataclass

from datachat.core.document import Document
from datachat.store.pinecone_store import PineconeStore
from datachat.store.vector_store import VectorStore
from datachat.core.config import Config
from datachat.core.models import OpenAIEmbedding, OpenAIInference

from langchain.memory import ConversationBufferWindowMemory


@dataclass
class Dataset:
    name: str
    index_name: str
    system_prompt: str


class DataChat:

    def __init__(
        self,
        config: Optional[Config] = None,
        memory_size: int = 3,  # Keep last 3 message pairs by default
    ):
        """Initialize DataChat with documents and system prompt.

        Args:
            config: Optional configuration for OpenAI and Pinecone
        """

        self.config = config or Config.load()
        self._datasets = {}

        print()
        self.vector_store = PineconeStore(self.config.pinecone)
        self.embedding_model = OpenAIEmbedding(self.config.openai)
        self.inference_model = OpenAIInference(self.config.openai)
        # Initialize simple window memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_size, return_messages=True
        )

    def register(
        self, dataset_name: str, documents: List[Document], system_prompt: str
    ):
        index_name = self._get_index_name(dataset_name)
        self._datasets[dataset_name] = Dataset(dataset_name, index_name, system_prompt)

        print()
        print("Upserting doucments in Pinecone db...")
        vectors = [
            (
                document.id,
                self.embedding_model.create_embedding(document.text),
                document.metadata,
            )
            for document in documents
        ]
        self.vector_store.upsert(index_name, vectors=vectors)
        print("Finished upserting doucments in Pinecone db")

        print("Waiting for vectors to be indexed...")
        self._wait_for_indexing()
        print("Vectors are now indexed and ready for querying")

    def generate_response(self, dataset_name, user_query: str, top_k: int = 100) -> str:
        """Generate a response based on the user query and relevant context.

        Args:
            user_query: User's question about the data

        Returns:
            Generated response from the chat model
        """
        # Get relevant context from vector store
        dataset: Dataset = self._datasets[dataset_name]
        query_vector = self.embedding_model.create_embedding(user_query)
        context = self.vector_store.search(dataset.index_name, query_vector, top_k)

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

        return self.inference_model.generate_response(
            context, user_query, dataset.system_prompt, history_text
        )

    def _wait_for_indexing(self, timeout: int = 20, initial_delay: int = 5):
        """Wait for vectors to be indexed and queryable"""
        # Initial delay to allow indexing to start
        sleep(initial_delay)

    def _get_index_name(self, dataset_name):
        return "datachat-" + dataset_name
