from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from . import crud, schemas

app = FastAPI()


@app.get("/")
def get() -> str:
    return ("This is an API for the IUM project."
            "See /docs to get more information about endpoints :)")


""" BASE MODEL """


@app.post("/base-model/predict")
def base_model_predict(data: schemas.BaseModelPredict) -> JSONResponse:
    prediction = crud.get_predictions_base_model(data=data, model=None)
    
    response = {
        "track_id": data.track_id,
        "prediction": prediction
    }
    return response


@app.post("/base-model/predictions")
def base_model_generate_playilst(file: UploadFile) -> JSONResponse:
    playlist = crud.generate_playlist_base_model(file=file)
    return playlist



""" ADVANCED MODEL """


@app.post("/model/predictions")
def model_predict(file: UploadFile) -> JSONResponse:
    playlist = crud.generate_playlist_advanced_model(file)
    return playlist


""" A/B EXPERIMENTS """

