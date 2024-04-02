import time
start = time.time()
import math
import pandas as pd
import numpy as np
from numpy import random
from numpy.core.umath_tests import inner1d
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,cohen_kappa_score,classification_report
from sklearn.decomposition import PCA

# from sklearn.metrics import plot_confusion_matrix
plt.style.use("seaborn")

np.random.seed(41)

def print_confusion_matrix(cm,num):
  plt.figure(figsize=(5,5))
  plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
  plt.title('Confusion matrix', size = 15)
  plt.colorbar()
  tick_marks = np.arange(2)
  plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 2)
  plt.yticks(tick_marks, ["0", "1"], size = 2)
  plt.tight_layout()
  plt.ylabel('Actual label', size = 15)
  plt.xlabel('Predicted label', size = 15)
  width, height = cm.shape
  for x in range(width):
   for y in range(height):
    plt.annotate(str(cm[x][y]), xy=(y, x), 
    horizontalalignment='center',
    verticalalignment='center')
  plt.tight_layout()
  plt.show()
  # plt.savefig('confusion_matrix_model_' + str(num+1))




def f1(particle, classifier, x_id):
	tp = []
	for i in range(len(x_id[particle])):
		if x_id[particle][i] > 0.5: tp.append(i)
	X_selected_features = X[:, tp]
	### Using 80 percent for training and 20 percent for validating ###
	X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.20, random_state = 0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	fitness = accuracy_score(y_test, y_pred)
	return fitness

def f2(particle, classifier, x_id):
	tp = []
	for i in range(len(x_id[particle])):
		if x_id[particle][i] > 0.5: tp.append(i)
	X_selected_features = X[:, tp]
	### Using 80 percent for training and 20 percent for validating ###
	X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size = 0.20, random_state = 0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	P = accuracy_score(y_test, y_pred)
	alpha = 0.1
	nf = len(tp)
	nt = len(X[0])
	fitness = alpha * (1 - P) + (1 - alpha) * (1 - nf / nt)
	return fitness

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def PSO(num_particles, num_iter, w, c1, c2, clf, ff):
	x_id, v_id, p_id, p_best = [], [], [], []
	for i in range(num_particles):
		tmp = []
		tmp2 = []
		tmp3 = []
		for d in range(dimensions):
			if d < dimensions - 35:
				tmp.append(random.rand() / 10)
				tmp2.append(random.rand())
			else:
				tmp.append(random.rand())
				tmp2.append(random.rand())
			tmp3.append(0)
		x_id.append(tmp)
		v_id.append(tmp3)
		p_id.append(tmp2)
		p_best.append(0)

	if clf == "NN": classifier = classifier1
	elif clf == 'RF': classifier = classifier2
	elif clf == 'SVM': classifier = classifier3
	elif clf == 'DTC': classifier = classifier4
	else: print("Please enter a valid classifier"), exit(0)

	k = 1
	x1 = []
	y1 = []
	l = []

	ll = []

	for i in range(num_particles):
		l.append([])
	while k <= num_iter:
		for particle in range(num_particles):
			if ff == 1: fitness = f1(particle, classifier, x_id)
			elif ff == 2: fitness = f2(particle, classifier, x_id)
			else: print('Please enter a valid fitness value'), exit(0)
			if fitness > p_best[particle]:
				p_best[particle] = fitness
				for i in range(len(x_id[particle])):
					p_id[particle][i] = x_id[particle][i]
			l[particle].append(p_best[particle])
			# print(fitness)
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
		# print(g_best)
		tmp1 = []
		tmp2 = []
		# for i in range(num_particles):
			# val = sigmoid(k)
			# if p_id[i][677] < 0.5: 
			# 	if val < 0.5: tmp1.append(val)
			# 	else: tmp1.append(1 - val)
			# else:
			# 	if val < 0.5: tmp1.append(1 - val)
			# 	else: tmp1.append(val)
			# if p_id[i][678] < 0.5: 
			# 	if val < 0.5: tmp2.append(val)
			# 	else: tmp2.append(1 - val)
			# else:
			# 	if val < 0.5: tmp2.append(1 - val)
			# 	else: tmp2.append(val)
			# tmp1.append(p_id[i][677])
			# tmp2.append(p_id[i][678])
			# tmp1.append(principalComponents)
		# plt.plot(tmp1, tmp2, 'ro')
		# plt.show()

		# x1.append(tmp1)
		# x2.append(tmp2)
		k = k + 1
	
	principalComponents = pca.fit_transform(p_id)
	print('g')
	for i in principalComponents: print(i)
	# print("YO")
	# for i in range(num_particles):
	# 	print(p_id[i][674: ])


	for i in range(num_particles):
		plt.plot(l[i], label = "Particle" + str(i + 1))
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')
	plt.title('Fitness vs Iterations for different particles')
	plt.legend()
	plt.show()

	# print(x1)
	# print(y1)



	return g_d

def test(g_d, clf):
	if clf == "NN": classifier = classifier1
	elif clf == 'RF': classifier = classifier2
	elif clf == 'SVM': classifier = classifier3
	elif clf == 'DTC': classifier = classifier4
	else: print("Please enter a valid classifier"), exit(0)
	tp = []
	for i in range(len(g_d)):
		if g_d[i] > 0.5: tp.append(i)

	X_selected_features = X[:, tp]
	classifier.fit(X_selected_features, y)

	### Using the trained classifier for testing the model on images of team members and classmates
	test_data = pd.read_csv('Test.csv')
	col_names = list(test_data.columns.values)
	dimensions = len(col_names) - 1
	y_test = test_data.iloc[:, dimensions].values
	X_test = test_data.iloc[:, :dimensions].values
	X_test_selected_features = X_test[:, tp]
	y_test_pred = classifier.predict(X_test_selected_features)
	# plot_confusion_matrix(clf, X_test_selected_features, y_test)  # doctest: +SKIP
	# plt.show()  # doctest: +SKIP
	cm = confusion_matrix(y_test, y_test_pred)
	print_confusion_matrix(cm, 0)

	print(classification_report(y_test, y_test_pred))
	kappa = cohen_kappa_score(y_test_pred,y_test)
	print("kappa = ", kappa)
	print("Accuracy Test = ", accuracy_score(y_test, y_test_pred))
	print()

### Iniitialising the classifiers ###
pca = PCA(n_components=2)
classifier1 = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(5, 2), random_state = 1)
classifier2 = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier3 = SVC(kernel = 'rbf',random_state = 0)
classifier4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

### Loading the CK+ dataset ###
data = pd.read_csv('data.csv')
col_names = list(data.columns.values)
dimensions = len(col_names) - 1
y = data.iloc[:, dimensions].values
X = data.iloc[:, :dimensions].values
cl = 'SVM'
feature_subset = PSO(30, 30, 0.1, 0.1, 0.1, cl, 2)
test(feature_subset, cl)

print("Execution time in minutes = ", (time.time() - start) / 60)