import pickle
import onnxruntime as ort
from pathlib import Path

MODELS_DIR = Path("./models")

def load_pickle_model(filename: str):
    with open(MODELS_DIR / filename, "rb") as f:
        return pickle.load(f)

def load_onnx_model(filename: str):
    onnx_model = str(MODELS_DIR / filename)
    sess = ort.InferenceSession(onnx_model)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name