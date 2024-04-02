import time
start = time.time()
import pandas as pd
import numpy as np
from numpy import random
from numpy.core.umath_tests import inner1d
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
np.random.seed(41)

### Loading the CK+ dataset ###
data = pd.read_csv('data.csv')
col_names = list(data.columns.values)
data = data.drop(col_names[: 674], axis = 1) # Dropping all non AU columns
col_names = list(data.columns.values)
dimensions = len(col_names) - 1
y = data.iloc[:, dimensions].values
X = data.iloc[:, :dimensions].values

### PARTICLE SWARM OPTIMIZATION to find feature subset ###
num_particles = 30
w = 0.5
c1 = 0.8
c2 = 0.9

x_id, v_id, p_id, p_best = [], [], [], []
for i in range(num_particles):
	tmp = []
	tmp2 = []
	for d in range(dimensions):
		tmp.append(random.rand())
		tmp2.append(random.rand())
	x_id.append(tmp)
	v_id.append([0] * dimensions)
	p_id.append(tmp2)
	p_best.append(0)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(5, 2), random_state=1)

k = 1
while k < 31:
	for particle in range(num_particles):
		tp = []
		for i in range(len(x_id[particle])):
			if x_id[particle][i] > 0.5: tp.append(i)
		X_selected_features = X[:, tp]
		### Using 80 percent for training and 20 percent for validating ###
		X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.20, random_state = 0)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		fitness = accuracy_score(y_test, y_pred)
		if fitness > p_best[particle]: 
			p_best[particle] = fitness
			for i in range(len(x_id[particle])):
				p_id[particle][i] = x_id[particle][i]
	g_best = 0
	for particle in range(num_particles):
		if p_best[particle] > g_best:
			g_best = p_best[particle]
			g_d = []
			for i in range(len(p_id[particle])): g_d.append(p_id[particle][i])
	for i in range(num_particles):
		for d in range(dimensions):
			r1 = random.rand()
			r2 = random.rand()
			v_id[i][d] = w * v_id[i][d] + c1 * r1 * (p_id[i][d] - x_id[i][d]) + c2 * r2 * (g_d[d] - x_id[i][d])
			x_id[i][d] = x_id[i][d] + v_id[i][d]
	k = k + 1

### Now using only the selected features to train the classifier ###
tp = []
for i in range(len(g_d)):
	if g_d[i] > 0.5: tp.append(i)

X_selected_features = X[:, tp]
classifier.fit(X_selected_features, y)

### Using the trained classifier for testing the model on images of team members and classmates
data = pd.read_csv('Test.csv')
col_names = list(data.columns.values)
data = data.drop(col_names[: 674], axis = 1) # Dropping all non AU columns
col_names = list(data.columns.values)
dimensions = len(col_names) - 1
y_test = data.iloc[:, dimensions].values
X_test = data.iloc[:, :dimensions].values
X_selected_features = X_test[:, tp]
y_pred = classifier.predict(X_selected_features)
print(y_pred)

print(y_pred[56], y_pred[55])

print(confusion_matrix(y_test, y_pred))
print("Accuracy = ", accuracy_score(y_test, y_pred))
print()

print("Execution time in seconds = ", time.time() - start)