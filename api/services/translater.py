import json
from typing import List
import requests


class Translater:
    
    url = 'http://localhost:5400/translate'
        
     
    def translate_text(self, text: str):
        myobj = {
            "q": text,
            "source": "fr",
            "target": "en",
            "format": "text",
            "alternatives": 1,
        }
        headers = {'Content-type': 'application/json'}

        response = requests.post(self.url, json=myobj, headers=headers)
        translated_text = json.loads(response.text)['translatedText']
        
        return translated_text
    
    def translate_texts(self, texts: List[str]):
        """
        Traduit une liste de textes.
        """
        resps = []
        for text in texts:
            if text is not None and len(text) > 0:
                resps.append(self.translate_text(text))
        return resps