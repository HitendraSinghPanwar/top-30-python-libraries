#numpy

import numpy as np

arr = np.array([1,2,3,4,5])
print("1d",arr)

arr1 = np.array([[1,2,3],
               [4,5,6]])
print("2d",arr1)


arr2 = np.zeros((2,2))
print("arr2",arr2)

arr3 = np.ones((2,4))
print("aar3",arr3)

arr4 = np.arange(0, 20, 5)
print("aar4",arr4)

arr5 = np.random.rand(2, 2)
print("arr5",arr5)

arr6 = np.array([[1,2,3],[4,5,6]])
print(arr6[1:2])
print("aar6",arr6[-1])
print(arr6.shape)

arr7 = arr6.reshape(3, 2)
print("arr7",arr7)

a = np.array([[1,2], [5,6]])
b = np.array([[4,5], [7,8]])

print("matrix multi",np.dot(a,b))
print("mean", np.mean(a))
print("var", np.var(b))

print("add",a+b)
print("sub",a-b)
print("multi",a*b)
print("divide",a/b)


data1 = np.genfromtxt(r"C:\Users\thaku\Downloads\students_scores.csv", delimiter=",", skip_header=1, usecols=(2, 3, 4))
clm = np.mean(data1, axis=0)
print("clm",clm)
print("data",data1)
print("sum",np.sum(data1))



#scikit learn

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

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
print("accuracy:",accuracy)


#tensorflow

import  tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], 28 * 28 )).astype('float32') / 255.0
test_images = test_images.reshape((test_images.shape[0], 28 * 28 )).astype('float32') / 255.0

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu", input_shape=(28 * 28)),
    Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc:", test_acc)


#pytorch

import torch
from torch import nn
import torch.nn.functional as F


class neuralnetwork(nn.Module):
    def __init__(self):
        super(neuralnetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

model = neuralnetwork()
print(model)



#openCV

import cv2

img = cv2.imread(r"C:\Users\thaku\Downloads\OIP (1).jpeg", cv2.IMREAD_COLOR)

image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

cv2.imshow("Original", image1)

cv2.waitKey(0)
cv2.destroyAllWindows()

#matplotlib

import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\thaku\Downloads\OIP (1).jpeg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Sample Image")
plt.axis("on")
plt.show()



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig.add_axes([0.6, 0.6, 0.25, 0.25])

ax1.plot([2, 3, 4, 5, 5, 6, 6],
         [1, 8, 4, 9, 3, 2, 8])
ax2.plot([1, 2, 3, 4, 5],
         [2, 3, 4, 5, 6])

plt.show()

#xgboost

import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_iris()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = xgb.XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
print("accuracy:",accuracy)


#spaCy’s

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("spaCy is an industrial-strength NLP library built for production use.")

for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.dep_}\t{token.ent_type_}")

print("\nNamed Entity:")
for ent in doc.ents:
    print(f"{ent.text}\t{ent.label_}")

# displacy.serve(doc, style='dep', host='localhost', port=8080)

#scipy

from scipy import linalg
import numpy as np

x = np.array([[16, 4], [100, 25]])

print("\nMatrix square root:\n", linalg.sqrtm(x))
print("\nMatrix exponential:\n", linalg.expm(x))
print("\nMatrix sine:\n", linalg.sinm(x))
print("\nMatrix cosine:\n", linalg.cosm(x))
print("\nMatrix tangent:\n", linalg.tanm(x))


#lightgbm

import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_iris()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = lgb.LGBMClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
print("accuracy:",accuracy)

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

#hugging face transformer

from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_auth_token=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", use_auth_token=False)

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2,)

story = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("generated text:\n",story)

#plotly

import plotly.express as px

df = px.data.iris()

fig = px.bar(df, y="sepal_length", x="sepal_width", color="species")

fig.show()

#seaborn

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset("iris")

sns.lineplot(x="sepal_length", y="sepal_width", data=data)

sns.set_style("dark")
plt.show()


#MLflow

import mlflow
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

with mlflow.start_run():
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

print("accuracy:", accuracy)

#Dask

import dask.dataframe as dd

df = dd.read_csv(r"C:\Users\thaku\Downloads\dataset.csv")

numeric_df = df.select_dtypes(include='number')

mean_value = numeric_df.mean().compute()

print("average value:\n", mean_value )

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
