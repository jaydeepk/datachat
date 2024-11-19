
import os
import pytest
from dotenv import load_dotenv

from datachat.core.config import Config, Environment

def pytest_sessionstart(session):
    """Load test configuration at session start"""
    try:
        Config.load(Environment.TEST)  # Validates test configuration
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Test configuration error: {e}")