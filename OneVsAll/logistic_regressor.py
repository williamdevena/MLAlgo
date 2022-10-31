import numpy as np
from scipy.special import expit
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

class LogisticRegressor():
	    
	def __init__(self, lr):
		self.w = np.array([0.1, 0.4, 0.6])
		self.lr = lr

	def fit(self, X_train, Y_train, epochs):
		self.w = np.zeros(X_train[0].shape[0])	
		if X_train.shape[0]!=Y_train.shape[0]:
			print("X_train Y_train di dimensioni diverse")			
		for epoch in range(epochs):
			for index in range(len(X_train)):
				self.gradientDescent(X_train[index], Y_train[index])
				
		return self				

	def gradientDescent(self, x, y):
		g_w = (-y*x)/(1+(np.exp(y*(np.dot(self.w, x)))))
		self.w = self.w - self.lr*g_w

	def predict(self, x):
		w_grande = [self.w]*len(x)
		x = list(map(np.dot, x, w_grande))
		Y_pred = list(map(self.sigmoid, x))	
		Y_pred=np.array(Y_pred)
		Y_pred = np.where(Y_pred>=0.5, 1, -1)

		return Y_pred

	def sigmoid(self, x):
		return (1/(1 + np.exp(-x)))

	def lossFunction(self, x, y):
		return np.log(1 + np.exp(-y*np.dot(self.w, x)))
	
	
	def accuracy(self, Y_pred, Y_test):
		return sum(Y_pred==Y_test)/len(Y_test)