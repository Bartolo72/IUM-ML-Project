import json
import os
from typing import List

import pandas as pd

dirname = os.path.dirname(__file__)
input_path = os.path.join(dirname,
                          "data/"
                          "full_model_input.json")
output_path = os.path.join(dirname,
                           "data/"
                           "result.json")
ranking_path = os.path.join(dirname,
                            "data/"
                            "example_ranking.json")

with open(input_path, 'r') as f:
    input_data = json.load(f)

data_weeks: List[pd.DataFrame] = []
for i, week in enumerate(input_data):
    if i == 0:
        continue
    df = pd.DataFrame(week)
    data_weeks.append(df)

with open(output_path, 'r') as f:
    output_data = json.load(f)


def generate_ranking(output_data: List[float],
                     data_weeks: List[pd.DataFrame]) -> None:
    columns_to_drop = ["track_id", "popularity", "duration_ms", "explicit",
                       "danceability", "energy", "key", "loudness",
                       "speechiness", "acousticness", "instrumentalness",
                       "liveness", "valence", "tempo", "release_date_year",
                       "release_date_week", "likes", "number_of_skips"]

    for i, df in enumerate(data_weeks):
        tmp_df = df.copy()
        tmp_df.drop(columns=columns_to_drop, inplace=True)
        tmp_df = tmp_df.values.tolist()
        tmp_df_2 = []
        for row in tmp_df:
            tmp_df_2.extend(row)
        data_weeks[i] = tmp_df_2

    for i in range(len(output_data)):
        output_data[i] = (output_data[i] + data_weeks[0][i] +
                          data_weeks[1][i] + data_weeks[2][i])

    ranking = sorted(range(len(output_data)),
                     key=lambda i: output_data[i])
    ranking.reverse()

    for i, idx in enumerate(ranking[:50]):
        print(f"{i}: {idx} - {output_data[idx]}")

    with open(ranking_path, 'w+') as f:
        f.write(json.dumps(ranking, indent=4))


generate_ranking(output_data, data_weeks)
