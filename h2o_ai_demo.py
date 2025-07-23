import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_iris
import pandas as pd

h2o.init()

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

hf = h2o.H2OFrame(df)
hf['target'] = hf['target'].asfactor()

train, test = hf.split_frame(ratios=[0.8])

aml = H2OAutoML(max_models=5, seed=1)
aml.train(y='target', training_frame=train)

lb = aml.leaderboard
print(lb)

preds = aml.predict(test)
print(preds.head())

h2o.cluster().shutdown(prompt=False)
