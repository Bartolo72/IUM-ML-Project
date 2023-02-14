import json
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import UploadFile
from joblib import load
from sklearn.linear_model import LinearRegression
from fastapi.encoders import jsonable_encoder

from .models.adv_model.adv_model_pred import predict
from .schemas import BaseModelPlaylist, BaseModelPredict


""" UTILS """

def save_playlist(input_data, results, is_base=False):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = os.path.dirname(__file__)
    save_path = os.path.join(dirname, "database/base_model/") if is_base else os.path.join(dirname, "database/adv_model/")

    dir_path = os.path.join(save_path, f"{timestamp}/")[1:]
    if not os.path.exists(dir_path):
        os.umask(0)
        os.mkdir(dir_path, mode=0o777)

    with open(f"{dir_path}input.json", "w") as f:
        json.dump(input_data, f, indent=4)

    # output
    with open(f"{dir_path}output.json", "w") as f:
        f.write(json.dumps(results, indent=4))


""" BASE MODEL """


def get_predictions_base_model(data: BaseModelPredict, model: LinearRegression | None ) -> int:
    if not model:
        model = load(filename="./microservice/models/base_model/base_model.joblib")
    x = pd.DataFrame([{
        "play_count_week_1": data.play_count_week_1,
        "play_count_week_2": data.play_count_week_2,
        "play_count_week_3": data.play_count_week_3,
        "play_count_week_4": data.play_count_week_4
    }])
    output = model.predict(x)
    prediction = int(output)
    return prediction


def generate_playlist_base_model(file: UploadFile) -> List:
    model = load(filename="./microservice/models/base_model/base_model.joblib")

    input_data = json.load(file.file)
    playlist = []
    tracks = []
    
    for sample in input_data:
        sample = BaseModelPredict(**sample)
        tracks.append(jsonable_encoder(sample))
        prediction = get_predictions_base_model(data=sample, model=model)
        track = {
            "track_id": sample.track_id,
            "prediction": prediction
        }
        playlist.append(track)

    with open('./microservice/unique_ids.json', 'r') as file_handler:
        unique_ids = json.loads(file_handler.read())
        playlist_tracks_ids = [track["track_id"] for track in tracks]
        for track_id in unique_ids:
            if track_id not in playlist_tracks_ids:
                track = {
                    "track_id": track_id,
                    "prediction": 0
                }
                playlist.append(track)

    results = pd.DataFrame(playlist)
    results.sort_values(by=["track_id"], inplace=True, ascending=True)
    results.drop(["track_id"], inplace=True, axis=1)
    predictions = results.values.tolist()
    
    tmp_df = []
    for row in predictions:
        tmp_df.extend(row)
    predictions = tmp_df

    save_playlist(input_data=tracks, results=predictions, is_base=True)
    return predictions


""" ADVANCED MODEL """


def generate_playlist_advanced_model(file: UploadFile):
    input_data = json.load(file.file)
    results = predict(input_data=input_data, model_rel_path="./microservice/models/adv_model/adv_model.tar")

    save_playlist(input_data=input_data, results=results)
    return results
