from pydantic import BaseModel, Field
from typing import Dict


class PredictionResponse(BaseModel):
    """
    Response model for the prediction endpoint.

    Ensures consistent output format and generates API documentation.
    """

    predicted_category: str = Field(
        ...,
        description="The predicted insurance premium category",
        examples=["High"]
    )

    confidence: float = Field(
        ...,
        description="Model's confidence score for the predicted class (0 to 1)",
        examples=[0.8432],
        ge=0,
        le=1
    )

    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all possible classes",
        examples=[{"Low": 0.01, "Medium": 0.15, "High": 0.84}]
    )