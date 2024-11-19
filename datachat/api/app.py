from fastapi import FastAPI
from .routes import router


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="DataChat API",
        description="API for json data based chat interactions",
        version="1.0.0",
        debug=True,
    )

    app.include_router(router, prefix="/api/v1")

    return app
