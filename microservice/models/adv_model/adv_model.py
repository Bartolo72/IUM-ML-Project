import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data

sns.set_theme(style="darkgrid")

dirname = os.path.dirname(__file__)
train_path = os.path.join(dirname,
                          "data/training_data_by_weeks/")
model_path = os.path.join(dirname, "microservice/adv_model/adv_model.tar")

losses_path = os.path.join(dirname, "graphs/adv_model_losses.png")


# Model
class NeuralNet(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_hidden1: int,
                 num_hidden2: int,
                 num_outputs: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_outputs)
        self.act_fn = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        return x


def main():
    global dirname
    global train_path
    global model_path
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

    # Read data
    data_weeks: List[pd.DataFrame] = []
    for file in os.listdir(train_path):
        file_path = os.path.join(train_path, file)
        df = pd.read_json(file_path)
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
    for i in range(len(data_weeks) - 4):
        tmp_df_0 = data_weeks[i].copy().values.tolist()
        tmp_df_1 = data_weeks[i + 1].drop(columns=columns_to_drop,
                                          inplace=False).values.tolist()
        tmp_df_2 = data_weeks[i + 2].drop(columns=columns_to_drop,
                                          inplace=False).values.tolist()
        tmp_df_3 = data_weeks[i + 3].drop(columns=columns_to_drop,
                                          inplace=False).values.tolist()
        training_data.append(tmp_df_0 + tmp_df_1 + tmp_df_2 + tmp_df_3)

    for i in range(len(training_data)):
        tmp = []
        for row in training_data[i]:
            tmp.extend(row)
        training_data[i] = tmp

    training_labels = []
    for i in range(len(data_weeks) - 4):
        tmp_df = data_weeks[i + 4].drop(columns=["popularity", "duration_s",
                                                 "explicit", "danceability",
                                                 "energy", "key", "loudness",
                                                 "speechiness", "acousticness",
                                                 "instrumentalness",
                                                 "liveness", "valence",
                                                 "tempo", "release_date_year",
                                                 "release_date_week", "likes",
                                                 "number_of_skips"],
                                        inplace=False)
        tmp_df = tmp_df.values.tolist()
        tmp_df_2 = []
        for row in tmp_df:
            tmp_df_2.extend(row)
        training_labels.append(tmp_df_2)

    test_index = random.randint(0, len(training_data) - 1)
    test_data = [training_data.pop(test_index)]
    test_labels = [training_labels.pop(test_index)]

    train_data_tensor = torch.FloatTensor(training_data)
    train_labels_tensor = torch.FloatTensor(training_labels)

    test_data_tensor = torch.FloatTensor(test_data)
    test_labels_tensor = torch.FloatTensor(test_labels)

    train_dataset = data.TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=128,
                                   shuffle=False)

    # Choose hyperparameters

    model = NeuralNet(num_inputs=109917,
                      num_hidden1=2048,
                      num_hidden2=1024,
                      num_outputs=4071).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0005,
                                 weight_decay=0.0001)

    loss_module = nn.L1Loss()

    model.train()

    losses = []
    epoch_num = 4096
    for epoch in range(epoch_num):
        for data_inputs, data_labels in train_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs)

            loss = loss_module(preds, data_labels.float())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        losses.append(loss.item())
        print(f"[{epoch + 1}/{epoch_num}] loss: {loss.item():.3}")

    state_dict = model.state_dict()
    torch.save(state_dict, model_path)

    torch.set_printoptions(threshold=10000)

    model.eval()
    print(model(test_data_tensor.to(device)).round())
    print(test_labels_tensor)

    plt.plot(losses)
    plt.savefig(losses_path)


if __name__ == "__main__":
    main()
