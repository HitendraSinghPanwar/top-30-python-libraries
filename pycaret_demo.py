from pycaret.classification import *
from pycaret.datasets import get_data

data = get_data('iris')
clf_setup = setup(data, target='species')
best_model = compare_models()
dt_model = create_model('dt')
tuned_dt = tune_model(dt_model)
plot_model(tuned_dt, plot='confusion_matrix')
predictions = predict_model(tuned_dt, data=data)
save_model(tuned_dt, 'decision_tree_model')