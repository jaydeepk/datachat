
import os
import pytest
from dotenv import load_dotenv

def pytest_sessionstart(session):
    """
    Called before test session starts.
    Load test environment variables.
    """
    # First try to load from tests/.env.test
    env_file = os.path.join(os.path.dirname(__file__), '.env.test')
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        # Fallback to root .env file
        load_dotenv()
    
    # Verify required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_ENVIRONMENT'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing required test environment variables: {', '.join(missing)}\n"
            "Please create a .env.test file in the tests directory with these variables."
        )