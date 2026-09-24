from pydantic import BaseModel, Field
from typing import Literal

class TextDetectionRequest(BaseModel):
    text: str = Field(..., description="The text to analyze")

class TextDetectionResponse(BaseModel):
    label: Literal["ai", "human"]
    confidence: float
    model_version: str
