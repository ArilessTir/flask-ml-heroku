from sklearn.model_selection import train_test_split
from loadData import load_model_csv
import Config
import pickle


def train_save(model, name=Config.MODEL_NAME):
    X, y = load_model_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    pickle.dump(model, open(name, 'wb'))
    print(f'model sauvegarder sous le nom{name}')
