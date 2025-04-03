from fastapi import APIRouter
from pydantic import BaseModel
from services.model_loader import load_pickle_model
from services.predictor import predict_with_pickle
from services.postprocessing import PostProcessing

router = APIRouter(prefix="/knn", tags=["KNN"])

# Chargement du modèle KNN
knn_model = load_pickle_model("knn_model.pkl")

class InputData(BaseModel):
    features: list[float]

@router.post("/predict")
def predict_knn(data: InputData):
    prediction = predict_with_pickle(knn_model, data.features)
    return {"prediction": PostProcessing.post_process(prediction)}