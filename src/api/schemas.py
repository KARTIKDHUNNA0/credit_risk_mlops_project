from pydantic import BaseModel
from typing import Dict

class PredictRequest(BaseModel):
    data: Dict[str, float]  # single row of feature_name: value mapping
