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
data = pd.read_csv('dataset.csv')

col_names = list(data.columns.values)

data = data.drop(col_names[: 674], axis = 1) # Dropping all non AU columns

# data = data.sample(frac = 1).reset_index(drop = True) # Data shuffle

col_names = list(data.columns.values)

dimensions = len(col_names) - 1

print(dimensions)

y = data.iloc[:, dimensions].values
X = data.iloc[:, :dimensions].values

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

# X_train = X[: 576]
# y_train = y[: 576]
# X_test = X[576: ]
# y_test = y[576: ]

# print('NeuralNetwork', end = " -> ")
# from sklearn.neural_network import MLPClassifier
# classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred))

### PARTICLE SWARM OPTIMIZATION to find feature subset ###
num_particles = 30
w = 0.5
c1 = 0.8
c2 = 0.9

x_id = []
v_id = []
p_id = []
p_best = []
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

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

k = 1
while k < 151:
	for particle in range(num_particles):
		tp = []
		for i in range(len(x_id[particle])):
			if x_id[particle][i] > 0.5: tp.append(i)
		X_selected_features = X[:, tp]
		X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.25, random_state = 0)
		# X_train = X_selected_features[: 576]
		# y_train = y[: 576]
		# X_test = X_selected_features[576: ]
		# y_test = y[576: ]
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


### Now using only the selected features to classify ###


tp = []
for i in range(len(g_d)):
	if g_d[i] > 0.5: tp.append(i)

print('the selected', len(tp), 'feature columns are -> ', tp)

X_selected_features = X[:, tp]
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.25, random_state = 0)
# X_train = X_selected_features[: 576]
# y_train = y[: 576]
# X_test = X_selected_features[576: ]
# y_test = y[576: ]
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy = ', accuracy_score(y_test, y_pred))


print()
tp = []
for i in range(dimensions):
	cnt = 0
	for j in range(num_particles):
		if p_id[j][i] > 0.5: cnt -= -1
	if cnt > num_particles // 2: tp.append(i)
print('the selected', len(tp), 'feature columns are -> ', tp)

X_selected_features = X[:, tp]
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.25, random_state = 0)
# X_train = X_selected_features[: 576]
# y_train = y[: 576]
# X_test = X_selected_features[576: ]
# y_test = y[576: ]
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy = ', accuracy_score(y_test, y_pred))

print()


# for i in range(num_particles):
# 	for j in range(dimensions):
# 		if p_id[i][j] > 0.5: print(1, end = " ")
# 		else: print(0, end = " ")
# 	print()



print("Execution time in seconds = ", time.time() - start)




# the selected 20 feature columns are ->  [0, 1, 2, 4, 6, 7, 8, 9, 12, 13, 14, 16, 19, 24, 25, 26, 28, 29, 31, 33]
# accuracy =  1.0