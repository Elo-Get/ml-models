from typing import List
import numpy as np
from onnxruntime import InferenceSession
from sklearn.base import BaseEstimator
from services.preprocessing import PreProcessing
from services.distil_bert_handler import DistilBertHandler

preprocessing = PreProcessing("./models/vectorizer.pkl")
distilBertHandler = DistilBertHandler()
distilBertHandler.load_model("checkpoint_final.pth")

def predict_with_pickle(model: BaseEstimator, text: str, preprocess_func):
    processed_text = preprocess_func(text)
    return model.predict(processed_text).tolist()

def predict_with_onnx(sess: InferenceSession, input_name: str, output_name: str, texts: List[str]):
    translated_texts = preprocessing.translate_texts(texts)
    data = preprocessing.vectorize_texts(translated_texts).toarray().astype(np.float32)
    pred_onnx = sess.run([output_name], {input_name: data})[0]
    return pred_onnx

def predict_llm(texts: List[str]):
    translated_texts = preprocessing.translate_texts(texts)
    print("Translated texts:", translated_texts)
    tokens = distilBertHandler.preprocess_texts(translated_texts)
    res = distilBertHandler.make_inference(tokens)
    print(res)
    return res