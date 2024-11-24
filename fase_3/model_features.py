import numpy as np
from pydantic import BaseModel


class ModelParams(BaseModel):
    pclass: int
    name:str
    sex:str
    age:float
    sibsp:int
    parch:int
    fare:float
    cabin:str
    embarked:str
    ticket: str