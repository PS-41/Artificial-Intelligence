import time
start = time.time()
import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('dataset.csv')

col_names = list(data.columns.values)

# data = data.drop(col_names[: 674], axis = 1)

data = data.sample(frac = 1).reset_index(drop = True)

col_names = list(data.columns.values)

num_col = len(col_names)
# print(len(col_names))

y = data.iloc[:, num_col - 1].values
X = data.iloc[:, :num_col - 1].values

# X_train = X[: 576]
# y_train = y[: 576]
# X_test = X[576: ]
# y_test = y[576: ]


# print('DecisionTreeClassifier', end = " -> ")
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# print('RandomForestClassifier', end = " -> ")
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# PSO
from numpy import random


# position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1)**(bool(random.getrandbits(1))) * random.random()*50])

# print(position)


num_particles = 30





















# x_id = random.randint(2, size=(num_particles, num_col - 1))
x_id = []
v_id = []
p_id = []
p_best = []
for i in range(num_particles):
	tmp = []
	tmp2 = []
	for d in range(num_col - 1):
		tmp.append(random.rand())
		tmp2.append(random.rand())
	x_id.append(tmp)
	v_id.append([0] * num_col - 1)
	p_id.append(tmp2)
	p_best.append(0)


# x_id[0][0] = 7

# v_id[0][0] = 99

# print(x_id[0])
# print(v_id)
# print(p_id[0])
# print(p_best)


# v_id = random.randint(1, size=(num_particles, num_col - 1))
# p_id = random.randint(2, size=(num_particles, num_col - 1))
# p_best = random.randint(1, size = (num_particles))

# p_best[0] = 0.987

# print(p_best)

# for particle in range(num_particles):
# 		print(x_id[particle])

# print(type(X[0][0]))
# print(x_id[0])
# tp = []
# for i in range(len(x_id[0])):
# 	if x_id[0][i]: tp.append(i)

# print("tp = ", tp)

# X_selected_features = X[:, tp]

# print(X_selected_features)

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
# k = 1
# w = 0.5
# c1 = 0.8
# c2 = 0.9
# while k < 4:
# 	for particle in range(num_particles):
# 		tp = []
# 		for i in range(len(x_id[particle])):
# 			if x_id[particle][i] > 0.5: tp.append(i)
# 		X_selected_features = X[:, tp]
# 		X_train = X_selected_features[: 576]
# 		y_train = y[: 576]
# 		X_test = X_selected_features[576: ]
# 		y_test = y[576: ]
# 		classifier.fit(X_train, y_train)
# 		y_pred = classifier.predict(X_test)
# 		fitness = accuracy_score(y_test, y_pred)
# 		if fitness > p_best[particle]: 
# 			p_best[particle] = fitness
# 			for i in range(len(x_id[particle])):
# 				p_id[particle][i] = x_id[particle][i]
# 	g_best = 0
# 	g_d = []
# 	for particle in range(num_particles):
# 		if p_best[particle] > g_best:
# 			g_best = p_best[particle]
# 			for i in range(len(p_id[particle])): g_d.append(p_id[particle][i])
# 			# g_d = p_id[particle]
# 	print(g_best)
# 	print(g_d)
# 	for i in range(num_particles):
# 		for d in range(num_col - 1):
# 			r1 = random.rand()
# 			r2 = random.rand()
# 			v_id[i][d] = w * v_id[i][d] + c1 * r1 * (p_id[i][d] - x_id[i][d]) + c2 * r2 * (g_d[d] - x_id[i][d])
# 			x_id[i][d] = x_id[i][d] + v_id[i][d]
# 	print("g_d = ", g_d)
# 	k = k + 1
# print(g_best)
# print(g_d)


print(time.time() - start)