import json
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch

from .adv_model import NeuralNet, model_path


def input_data_to_tensor(input_data):
    data_weeks: List[pd.DataFrame] = []
    for week in input_data:
        df = pd.DataFrame(week)
        df.drop(columns=["track_id"], inplace=True)
        df["duration_ms"] = df["duration_ms"].div(1000).round(2)
        df.rename(columns={"duration_ms": "duration_s"}, inplace=True)
        data_weeks.append(df)

    training_data = []
    columns_to_drop = ["popularity", "duration_s", "explicit", "danceability",
                       "energy", "key", "loudness", "speechiness",
                       "acousticness", "instrumentalness", "liveness",
                       "valence", "tempo", "release_date_year",
                       "release_date_week"]

    tmp_df_0 = data_weeks[0].copy().values.tolist()
    tmp_df_1 = data_weeks[1].drop(columns=columns_to_drop,
                                  inplace=False).values.tolist()
    tmp_df_2 = data_weeks[2].drop(columns=columns_to_drop,
                                  inplace=False).values.tolist()
    tmp_df_3 = data_weeks[3].drop(columns=columns_to_drop,
                                  inplace=False).values.tolist()
    training_data.append(tmp_df_0 + tmp_df_1 + tmp_df_2 + tmp_df_3)

    for i in range(len(training_data)):
        tmp = []
        for row in training_data[i]:
            tmp.extend(row)
        training_data[i] = tmp

    # Input data
    input_tensor = torch.FloatTensor(training_data[0])
    return input_tensor

def predict(input_data, model_rel_path=None):
    if model_rel_path:
        model_path = model_rel_path

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(64)
        torch.cuda.manual_seed_all(64)
    else:
        device = torch.device("cpu")

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(64)
    np.random.seed(64)
    random.seed(64)
        
    input_tensor = input_data_to_tensor(input_data)

    model = NeuralNet(num_inputs=109917,
                    num_hidden1=2048,
                    num_hidden2=1024,
                    num_outputs=4071).to(device).float()

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    results = model(input_tensor.to(device))
    results = results.detach().cpu().numpy().tolist()

    return results

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    full_input_path = os.path.join(dirname,
                                   "data/"
                                   "full_model_input.json")
    result_path = os.path.join(dirname,
                               "data/"
                               "result.json")

    with open(full_input_path, 'r') as f:
        input_data = json.load(f)
    

    with open(result_path, 'w') as f:
        f.write(json.dumps(results, indent=4))
