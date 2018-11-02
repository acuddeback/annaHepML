import hep_ml

import numpy, pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('MiniBooNE_PID.txt', sep='\s\s*', skiprows=[0], header=None, engine='python')
labels = pandas.read_csv('MiniBooNE_PID.txt', sep=' ', nrows=1, header=None)
labels = [1] * labels[1].values[0] + [0] * labels[2].values[0]
data.columns = ['feature_{}'.format(key) for key in data.columns]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.5, test_size=0.5, random_state=42)

# example of training a network
from hep_ml.nnet import MLPClassifier
from sklearn.metrics import roc_auc_score

clf = MLPClassifier(layers=[5], epochs=500)
clf.fit(train_data, train_labels)

proba = clf.predict_proba(test_data)
print('Test quality:', roc_auc_score(test_labels, proba[:, 1]))

proba = clf.predict_proba(train_data)
print('Train quality:', roc_auc_score(train_labels, proba[:, 1]))