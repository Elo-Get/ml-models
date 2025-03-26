import os
import joblib
import requests
import json
import pandas as pd
from typing import Tuple
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

class PreProcessing:
    
    def __init__(self, vectorizer_path: str = "vectorizer.pkl"):
        self.vectorizer_path = vectorizer_path
        
        if os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("Vectorizer loaded from save.")
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            print("New vectorizer created.")
    
    def prepare_text(self, text: str):
        """
        Traduit un texte en anglais et le vectorise.
        """
        url = 'http://localhost:5400/translate'
        myobj = {
            "q": text,
            "source": "fr",
            "target": "en",
            "format": "text",
            "alternatives": 1,
        }
        headers = {'Content-type': 'application/json'}

        response = requests.post(url, json=myobj, headers=headers)
        translated_text = json.loads(response.text)['translatedText']
        
        return self.vectorizer.transform([translated_text])  # Correction : besoin d'une liste
    
    def get_dataset(self) -> Tuple[pd.DataFrame]:
        """
        Charge le dataset IMDB-Spoiler et vectorise le texte.
        """
        ds = load_dataset("bhavyagiri/imdb-spoiler")
        
        data_train = pd.DataFrame(ds['train']).dropna()
        data_test = pd.DataFrame(ds['validation']).dropna()

        if not os.path.exists(self.vectorizer_path):
            X_train = self.vectorizer.fit_transform(data_train['text'])
            joblib.dump(self.vectorizer, self.vectorizer_path)  # Sauvegarde du vectorizer
            print("Vectorizer trained and saved.")
        else:
            X_train = self.vectorizer.transform(data_train['text'])

        y_train = data_train['label']
        X_test = self.vectorizer.transform(data_test['text'])
        y_test = data_test['label']
        
        return X_train, y_train, X_test, y_test
