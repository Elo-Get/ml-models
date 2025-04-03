from fastapi import APIRouter
from services.model_loader import load_onnx_model
from services.predictor import predict_with_onnx
from services.postprocessing import PostProcessing
from routers.types import InputData

router = APIRouter(prefix="/mlp", tags=["Machine Learning"])

sess, input_name, output_name = load_onnx_model("mlp_model.onnx")

@router.post("/predict")
def predict_mlp(data: InputData):
    prediction = predict_with_onnx(sess, input_name, output_name, data.texts)
    return {"prediction": PostProcessing.post_process(prediction)}