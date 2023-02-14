import json
import os
from typing import List

import pandas as pd

dirname = os.path.dirname(__file__)
train_path = os.path.join(dirname,
                          "data/training_data_by_weeks/")
save_path_full = os.path.join(dirname,
                              "data/"
                              "full_model_input.json")
save_path_basic = os.path.join(dirname,
                               "data/"
                               "basic_model_input.json")

output_path = os.path.join(dirname,
                           "data/"
                           "example_output.json")

data_weeks: List[pd.DataFrame] = []
data_next: List[pd.DataFrame] = []
for i, file in enumerate(os.listdir(train_path)):
    if i == 4:
        file_path = os.path.join(train_path, file)
        df = pd.read_json(file_path)
        data_next.append(df)
        break
    file_path = os.path.join(train_path, file)
    df = pd.read_json(file_path)
    data_weeks.append(df)

with open(save_path_full, 'w+') as f:
    f.write(json.dumps([df.to_dict(orient="records") for df in data_weeks],
                       indent=4))

columns_to_drop = ["popularity", "duration_ms", "explicit", "danceability",
                   "energy", "key", "loudness", "speechiness",
                   "acousticness", "instrumentalness", "liveness",
                   "valence", "tempo", "release_date_year",
                   "release_date_week", "likes", "number_of_skips"]

for i, df in enumerate(data_weeks):
    data_weeks[i] = df.drop(columns=columns_to_drop, inplace=False)

with open(save_path_basic, 'w+') as f:
    f.write(json.dumps([df.to_dict(orient="records") for df in data_weeks],
                       indent=4))

output_data = data_next[0].copy()
output_data.drop(columns=[*columns_to_drop, "track_id"], inplace=True)
output_data = output_data.values.tolist()
tmp_df = []
for row in output_data:
    tmp_df.extend(row)
print(tmp_df)

with open(output_path, 'w+') as f:
    f.write(json.dumps(tmp_df, indent=4))
