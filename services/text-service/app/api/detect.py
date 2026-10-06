from fastapi import APIRouter, HTTPException
import structlog
from app.schemas import TextDetectionRequest, TextDetectionResponse
from app.core.model import detector

router = APIRouter()
logger = structlog.get_logger()


@router.post("", response_model=TextDetectionResponse)
async def detect_text(request: TextDetectionRequest):
    if not request.text.strip():
        logger.warning("empty_text_request")
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = detector.predict(request.text)
        logger.info("text_detected", label=result.label, confidence=result.confidence)
        return result
    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction."
        )
