#Optuna

import optuna
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = datasets.load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def objective(trial):
    c = trial.suggest_float("C", 1e-4, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    model = LogisticRegression(C=c, solver=solver, max_iter=200)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
final_model = LogisticRegression(**best_params, max_iter=200)
final_model.fit(x_train, y_train)
y_pred = final_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best Params:", best_params)
print("Final Accuracy:", accuracy)