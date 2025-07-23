import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = datasets.load_iris()
x = data.data
y = data.target
feature_names = data.feature_names

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", accuracy)

explainer = shap.KernelExplainer(log_reg.predict_proba, x_train)

x_test_subset = x_test[:30]
shap_values = explainer.shap_values(x_test_subset)
shap.summary_plot(shap_values, x_test_subset, feature_names=feature_names)
