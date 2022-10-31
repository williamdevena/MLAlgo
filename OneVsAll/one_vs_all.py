from logistic_regressor import LogisticRegressor
from housing_dataset import HousingDataset

import numpy as np

class OneVsAll():
	
	def __init__(self, num_classes, lr, epochs):
		self.num_classes = num_classes
		self.lr = lr
		self.epochs = epochs
		self.classifiers = [LogisticRegressor(lr=self.lr) for x in range(num_classes)]
		
	
	
	def computeSubsetDS(self, Y):
		return [np.where(Y==classe, 1, -1)  for classe in range(self.num_classes)]
	
	
	def split(self, S, train_size):
		S_train = [si[:int(len(S[0])*train_size)] for si in S]
		S_test = [si[int(len(S[0])*train_size):] for si in S]
		
		return S_train, S_test
	
	
	def fit(self, X_train, Y_train):
		S_train = self.computeSubsetDS(Y_train)
		
		for classifier, Si_train in zip(self.classifiers, S_train):
			classifier.fit(X_train, Si_train, self.epochs)
			
			
	
	def predict(self, X_test):
		S_pred = [classifier.predict(X_test) for classifier in self.classifiers]	
		Y_pred = np.argmax(S_pred, axis=0)
		
		return Y_pred
	
		
	def accuracy(self, Y_pred, Y_test):
		return sum(Y_pred==Y_test)/len(Y_test)
	
	
	
	
	