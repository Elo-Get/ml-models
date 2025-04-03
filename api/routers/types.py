from typing import List
from pydantic import BaseModel


class InputData(BaseModel):
    texts: List[str]