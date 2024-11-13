from datachat.models import InferenceModel
from datachat.store.vector_store import VectorStore
from datachat.config import Config
from datachat.models import OpenAIInference


class DataChat:

    def __init__(
        self,
        vector_store: VectorStore,
        system_prompt: str = None,
        inference_model: InferenceModel = None,
        config: Config = None,
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

    def generate_response(self, user_query: str, top_k: int = 100) -> str:
        """Generate a response based on the user query and relevant context.

        Args:
            user_query: User's question about the data

        Returns:
            Generated response from the chat model
        """
        context = self.vector_store.search(user_query, top_k)
        return self.inference_model.generate_response(context, user_query)
