from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import Config

int_pipe = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Standarize', StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

transformer = make_column_transformer(
    ('drop', Config.DROP),
    (cat_pipe, Config.CAT),
    (int_pipe, Config.INT)
)
