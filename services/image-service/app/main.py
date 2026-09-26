"""
main.py — Entrypoint for Standalone Image Detection Microservice

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
"""

from src.image_detector.api import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.image_service.app.main:app", host="0.0.0.0", port=8004, reload=True)
