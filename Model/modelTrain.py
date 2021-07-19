from pipeline import pipeline
from loadData import load_model_csv
from savemodel import train_save
import pandas as pd
from sklearn.model_selection import cross_validate, KFold

X, y = load_model_csv()

cv = KFold(n_splits=5)

score = cross_validate(pipeline, X, y,
                       cv=cv, n_jobs=-1, return_train_score=True)
train_accuracy = score['train_score'].mean()
test_accuracy = score['test_score'].mean()

print(f'Score sur le Train: {train_accuracy}\nScore sur le Test: {test_accuracy}')
print(pipeline)
ask_save = input("Voulez vous sauvegarder ce model?(Y/N): ")

if ask_save == "Y" or ask_save == "y":
    train_save(pipeline)