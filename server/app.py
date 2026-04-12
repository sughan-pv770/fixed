"""
server/app.py - Required by OpenEnv validator for multi-mode deployment.
Exports the FastAPI app and a main() entry point.
"""
import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401 - re-exported for OpenEnv validator

__all__ = ["app", "main"]


def main():
    """Entry point for [project.scripts] server = 'server.app:main'"""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    main()
