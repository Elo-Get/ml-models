from typing import List
import numpy as np

class PostProcessing:

    @staticmethod
    def post_process(predictions: np.ndarray) -> List[str]:
        
        def _post_process(prediction):
            print(type(prediction))
            if isinstance(prediction, np.ndarray) or isinstance(prediction, list):
                prediction = prediction[0].item()
                
            if prediction == 0:
                return "Pas de spoiler"
            elif prediction == 1:
                return "Spoiler détecté"
            else:
                raise ValueError("Prediction must be either 0 or 1.")
        
        response = []
        for prediction in predictions:
            response.append(_post_process(prediction))
        
        return response