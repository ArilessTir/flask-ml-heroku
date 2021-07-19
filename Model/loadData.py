import pandas as pd
from sklearn.preprocessing import LabelEncoder
import Config

def load_model_csv(path=Config.DATA_PATH, sep=';'):
    data = pd.read_csv(filepath_or_buffer=path, sep=sep)
    X = data.drop([Config.TARGET], axis=1)
    y = pd.Series(LabelEncoder().fit_transform(data[Config.TARGET]), name="Survived")
    return X, y
