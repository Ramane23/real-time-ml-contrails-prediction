from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from loguru import logger
import uvicorn

from src.prediction import Predictor, ContrailPredictorOutput

#Instantiate the Predictor class
predictor = Predictor.from_model_registry(
        model_type='lightgbm',
        status='production',
    )
#Instantiating the FastAPI class
app = FastAPI()

class PredictRequest(BaseModel):
    route: str

class PredictResponse(BaseModel):
    prediction: ContrailPredictorOutput

@app.get("/healthcheck")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """this endpoint will return the prediction of the contrail formation for a given route

    Args:
        request (PredictRequest): the request object containing the route

    Returns:
        PredictResponse: the response object containing the prediction
    """
    try:
        prediction : ContrailPredictorOutput = predictor.predict_contrail_formation(request.route)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return PredictResponse(prediction=prediction.to_dict())

if __name__ == "__main__":
    
    # Running the FastAPI app with uvicorn
    #uvicorn aims to be a lightning-fast ASGI server, implementing asynchronous programming in Python
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)