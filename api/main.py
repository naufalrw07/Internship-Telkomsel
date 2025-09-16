from fastapi import FastAPI, HTTPException
from schemas import PredictionRequest
from models_config import MODELS_KEBALEN, MODELS_GAYUNGAN
from services.prediction import make_prediction, save_predictions

app = FastAPI()

@app.get("/")
def root():
    return {"message": "server is running"}

@app.post("/predict-kebalen")
def predict_kebalen(request: PredictionRequest):
    try:
        result = make_prediction(
            MODELS_KEBALEN,      # models_dict
            "kebalen",           # lokasi
            request.room,        # room
            request.duration_hours  # duration
        )
        save_predictions("kebalen", result["room"], result["predictions"])
        return {"prediction_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-gayungan")
def predict_gayungan(request: PredictionRequest):
    try:
        result = make_prediction(
            MODELS_GAYUNGAN,     # models_dict
            "gayungan",          # lokasi
            request.room,        # room
            request.duration_hours  # duration
        )
        save_predictions("gayungan", result["room"], result["predictions"])
        return {"prediction_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
