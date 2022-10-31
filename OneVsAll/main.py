from OneVsAll import OneVsAll
from housing_dataset import HousingDataset
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
 
	
def main():
	num_classes=3
	classifier = OneVsAll(num_classes=num_classes, lr=0.5, epochs=100)
	ds = HousingDataset(num_classes)
	classifier.fit(ds.X_train, ds.Y_train)
	
	Y_pred = classifier.predict(ds.X_test)
	print(classifier.accuracy(Y_pred, ds.Y_test))
	
	classifier2 = OneVsRestClassifier(LogisticRegression(random_state=0, solver='newton-cg'))
	classifier2.fit(ds.X_train, ds.Y_train)
	#Y_pred2=classifier2.predict(X_test)
	print('Accuracy OneVsAll Sklearn:', classifier2.score(ds.X_test, ds.Y_test, sample_weight=None))
	
	
	
if __name__ == '__main__':
	main()