from datachat.api.app import create_app
from datachat.core.registry import DocumentRegistry
from .conf_session_document import ConfSessionDocument

# Register document implementation
registry = DocumentRegistry.get_instance()
registry.register("conf-session", ConfSessionDocument)

# Create FastAPI app
app = create_app()