
#catboost

from catboost import CatBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_iris()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6)
model.fit(x_train, y_train)

y_pred_proba = model.predict_proba(x_test)
y_pred = y_pred_proba.argmax(axis=1)


accuracy = accuracy_score(y_pred, y_test)
print("accuracy:",accuracy)
