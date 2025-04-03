from fastapi import APIRouter
from routers.types import InputData
from services.model_loader import load_onnx_model
from services.predictor import predict_with_onnx
from services.postprocessing import PostProcessing

router = APIRouter(prefix="/random-forst", tags=["Machine Learning"])

sess, input_name, output_name = load_onnx_model("random_forest_model.onnx")

@router.post("/predict")
def predict_mlp(data: InputData):
    prediction = predict_with_onnx(sess, input_name, output_name, data.texts)
    return {"prediction": PostProcessing.post_process(prediction)}