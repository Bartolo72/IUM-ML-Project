from pydantic import BaseModel
from typing import List


class BaseModelPredict(BaseModel):
    track_id: str
    play_count_week_1: int
    play_count_week_2: int
    play_count_week_3: int
    play_count_week_4: int
    play_count: int | None

class BaseModelPlaylist(BaseModel):
    tracks: List[BaseModelPredict]
