import numpy as np
from math import sqrt
from scipy.spatial import distance
from sklearn import metrics

class KNN():
	def __init__(self, k, distance="euclidean"):
		self.k = k
		self.distance = distance
		self.X_train=np.array([])
		self.Y_train=np.array([])

	def fit(self, X_train, Y_train):
		if not isinstance(X_train, np.ndarray):
			X_train = np.array(X_train)
		if X_train.shape[0]!=Y_train.shape[0]:
			print("Fit error, X_train and Y_train are of different shapes: " + str(X_train.shape) + " " + str(Y_train.shape))
			return False
		self.X_train = X_train
		self.Y_train = Y_train

		return self.X_train, self.Y_train

	def calculateAccuracy(self, Y_pred, Y_test):
		return sum(np.equal(Y_pred, Y_test))/Y_test.shape[0]

	def predict(self, point):
		distances = [(self.euclideanDistance(point, self.X_train[index]), self.Y_train[index]) for index in range(len(self.X_train))]
		distances = sorted(distances, key = lambda x: x[0])
		neighbours = distances[:self.k]
		neighbours = [x[1] for x in neighbours]
		unique, counts = np.unique(neighbours, return_counts=True)
		neighbours = dict(zip(unique, counts))

		return max(neighbours, key=neighbours.get)

	def euclideanDistance(self, point1, point2):
		return distance.euclidean(point1, point2)

	def compare(self, knn2, X_test, Y_test):
		#Y_pred=knn2.predict(X_test)
		
		Y_pred1 = [self.predict(point) for point in X_test]
		Y_pred2 = [knn2.predict(point) for point in X_test]
		
		print()
		
		
		accuracy1 = self.calculateAccuracy(Y_pred1, Y_test)
		accuracy2 = metrics.accuracy_score(Y_test, Y_pred)
		print("Accuracy of knn: " + str(accuracy1))
		print("Accuracy of knn2: " + str(accuracy2))


	def findBestK(self, X_test, Y_test, k_limit):
		max_acc=0
		best_k=1
		for k in range(2, k_limit):
			self.k=k
			acc = self.calculateAccuracy(X_test, Y_test)
			if acc>max_acc:
				max_acc = acc
				best_k = k
		print("Best k: " + str(best_k))
		print("Best accuracy: " + str(max_acc))

		return best_k, max_acc












