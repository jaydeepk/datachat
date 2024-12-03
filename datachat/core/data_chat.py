from asyncio import sleep
import logging
from typing import Dict, List, Optional

from attr import dataclass

from datachat.core.document import Document
from datachat.store.pinecone_store import PineconeStore
from datachat.store.vector_store import VectorStore
from datachat.core.config import Config
from datachat.core.models import OpenAIEmbedding, OpenAIInference
from datachat.core.dataset_repository import Dataset, DatasetRepository

from langchain.memory import ConversationBufferWindowMemory


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
        self.dataset_repo = DatasetRepository()

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

        self.dataset_repo.upsert_dataset(
            Dataset(dataset_name, index_name, system_prompt)
        )

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
        """Generate a response for a dataset based on the user query and relevant context.

        Args:
            user_query: User's question about the data

        Returns:
            Generated response from the chat model
        """
        # Get relevant context from vector store
        dataset: Dataset = self.dataset_repo.get_dataset(dataset_name)
        if not dataset:
            raise Exception(f"Dataset {dataset_name} not found")

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

    def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset from both SQLite and Pinecone

        Args:
            dataset_name: Name of the dataset to delete

        Raises:
            ValueError: If dataset doesn't exist
            Exception: If deletion fails
        """
        # Check if dataset exists
        dataset: Dataset = self.dataset_repo.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        try:
            # Delete from Pinecone first
            self.vector_store.delete(dataset.index_name)

            # Delete from SQLite
            self.dataset_repo.delete_dataset(dataset_name)

            logging.info(f"Successfully deleted dataset '{dataset_name}'")

        except Exception as e:
            logging.error(f"Failed to delete dataset '{dataset_name}': {str(e)}")
            raise

    def _wait_for_indexing(self, timeout: int = 20, initial_delay: int = 5):
        """Wait for vectors to be indexed and queryable"""
        # Initial delay to allow indexing to start
        sleep(initial_delay)

    def _get_index_name(self, dataset_name):
        return "datachat-" + dataset_name
