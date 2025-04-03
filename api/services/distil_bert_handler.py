from typing import List
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from pathlib import Path

class DistilBertHandler:
        
    def __init__(self):
        
        self.MODELS_DIR = Path("./models")
        self.MODEL_NAME = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=2)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
    
    def load_model(self, filename: str):
        checkpoint = torch.load(self.MODELS_DIR / filename, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def preprocess_texts(self, texts: List[str], max_length: int = 512):
        encodings = self.tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        return {key: tensor.to(self.device) for key, tensor in encodings.items()} 

    def make_inference(self, inputs: dict):
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy()  