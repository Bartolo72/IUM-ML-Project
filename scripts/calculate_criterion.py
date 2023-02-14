import json
import os
from typing import List

dirname = os.path.dirname(__file__)
ranking_path_1 = os.path.join(dirname,
                              "data/"
                              "example_ranking.json")
ranking_path_2 = os.path.join(dirname,
                              "data/"
                              "example_ranking_2.json")

with open(ranking_path_1, 'r') as f:
    ranking_1 = json.load(f)

with open(ranking_path_2, 'r') as f:
    ranking_2 = json.load(f)


def calculate_criterion(ranking_1: List[int], ranking_2: List[int]) -> float:
    criterion = 0
    count_penalty = 0
    for i in range(50):
        index_1 = i + 1
        index_2 = ranking_2.index(ranking_1[i]) + 1
        if index_2 == index_1:
            count_penalty += 1
        criterion += ((index_1 - index_2) ** 2) / (i + 1)
    if count_penalty < 40:
        criterion *= 10
    return criterion


print(calculate_criterion(ranking_1, ranking_2))
