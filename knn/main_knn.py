from knn import KNN
from iris_dataset import IrisDataset
from sklearn.neighbors import KNeighborsClassifier


def main():
	k=4
	ds = IrisDataset()
	knn = KNN(k=k, distance="euclidean")

	X_train=ds.points[:100]
	Y_train=ds.labels[:100]
	X_test=ds.points[100:]
	Y_test=ds.labels[100:]

	knn.fit(X_train, Y_train)

	knn2 = KNeighborsClassifier(k)
	knn2.fit(X_train, Y_train)

	knn.compare(knn2, X_test, Y_test)
	knn.findBestK(X_test, Y_test, k_limit=10)
	
	Y_pred = [knn.predict(point) for point in X_test]
	print(knn.calculateAccuracy(Y_pred, Y_test))
	
	
	
	
	
if __name__ == '__main__':
	main()


