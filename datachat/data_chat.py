from datachat.models import InferenceModel
from datachat.store.vector_store import VectorStore
from datachat.config import Config
from datachat.models import OpenAIInference

from langchain.memory import ConversationBufferWindowMemory


class DataChat:

    def __init__(
        self,
        vector_store: VectorStore,
        system_prompt: str = None,
        inference_model: InferenceModel = None,
        config: Config = None,
        memory_size: int = 3  # Keep last 3 message pairs by default
    ):
        """Initialize DataChat with configurable parameters.

        Args:
            vector_store: Vector store for searching embeddings
            system_prompt: Custom system prompt for the chat model
            model: model to use for generating embeddings and completions
        """
        self.config = config if config is not None else Config.load()
        self.vector_store = vector_store
        self.inference_model = (
            inference_model
            if inference_model is not None
            else OpenAIInference(self.config.openai, system_prompt)
        )
        
        # Initialize simple window memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_size,
            return_messages=True
        )

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
        history_text = "\n".join([
            f"Human: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
            for msg in history
        ])
            
        return self.inference_model.generate_response(context, user_query, history_text)
    

