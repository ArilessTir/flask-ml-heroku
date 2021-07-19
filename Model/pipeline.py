import preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor.transformer),
    ('model', LogisticRegression())
])
