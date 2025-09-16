from pydantic import BaseModel

class PredictionRequest(BaseModel):
    room: str
    duration_hours: int