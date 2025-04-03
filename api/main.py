from routers import mlp, random_forest, svm, llm
from fastapi import FastAPI
from services.preprocessing import PreProcessing

app = FastAPI(title="ML Prediction API", description="API for predicting with ONNX and Pickle models", version="1.0")

# Inclure les routers pour chaque modèle
# app.include_router(knn.router)
app.include_router(llm.router)
app.include_router(mlp.router)
app.include_router(svm.router)
app.include_router(random_forest.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8420)