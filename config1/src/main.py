from fastapi import FastAPI
import numpy as np
from model import ModelServer

app = FastAPI()
model_server = ModelServer(path=model_path)

@app.post("/predict")
def predict(input_nda: np.ndarray):
    output_nda = model_server.make_prediction_unbatched(input_nda)
    return {"message": output_nda}