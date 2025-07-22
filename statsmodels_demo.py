import statsmodels.api as sm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = datasets.load_iris()
x = data.data[y := data.target]

x = x[y != 2][:, :2]
y = y[y != 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

model = sm.Logit(y_train, x_train)
result = model.fit(method="bfgs", maxiter=100)

y_pred_prob = result.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)
print(result.summary())
