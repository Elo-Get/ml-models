from fastapi import APIRouter
from routers.types import InputData
from services.model_loader import load_onnx_model
from services.predictor import predict_llm
from services.postprocessing import PostProcessing

router = APIRouter(prefix="/llm", tags=["LLM"])

@router.post("/predict")
def predict_mlp(data: InputData):
    predictions = predict_llm(data.texts)
    return {"prediction": PostProcessing.post_process(predictions)}