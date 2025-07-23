from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=data.feature_names,
    class_names=data.target_names,
    mode='classification'
)

exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=4)

print("Prediction probabilities:", model.predict_proba([X_test[0]]))
print("LIME Explanation:")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight}")
