from typing import Literal
from pydantic import BaseModel, Field


class TextDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyze")


class TextDetectionResponse(BaseModel):
    label: Literal["ai", "human"]
    confidence: float
    model_version: str
