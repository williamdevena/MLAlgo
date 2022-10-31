import numpy as np
import matplotlib.pyplot as plt

class Perceptron():

	def __init__(self):
		self.w=np.zeros(3)
		pass

	def fit(self, epochs, points, labels):
		self.w = np.zeros(points[0].shape[0])
		errors = []	    
		for epoch in range(epochs):
			print("EPOCA: " + str(epoch))
			error = 0
			for index in range(len(points)):
				point = points[index]
				label = labels[index]            
				out = np.dot(point, self.w)
				out = np.where(out >= 0, 1, -1)
				if out != label:
					error += 1
					self.w = self.w + (label * point)
			errors.append(error)
			if error==0:
				break
		figure, axis = plt.subplots(1, 2)
		axis[0].set_title("Misclassification error")
		axis[0].plot(errors)
		axis[1].set_title("Points and Perceptron")
		self.visualize(axis[1], points, labels)
		plt.show()

		return self.w

	def predict(self, point):
		if point.shape!=self.w.shape:
		 	print("Dimensioni del vettore di input sbagliate")
		out = np.dot(point, self.w)
		out = np.where(out >= 0, 1, -1)

		return out

	def visualize(self, axis, points, labels):
		if self.w.shape[0] != 3:
			print("per visualizzare deve avere dim=3")
		cx = self.w[1] 
		cy = self.w[2]
		cb = self.w[0]
		x1 = [points[index][1] for index in range(len(points)) if labels[index]==-1]
		y1 = [points[index][2] for index in range(len(points)) if labels[index]==-1]
		x2 = [points[index][1] for index in range(len(points)) if labels[index]==1]
		y2 = [points[index][2] for index in range(len(points)) if labels[index]==1]
		axis.scatter(x1,y1, marker="o")
		axis.scatter(x2,y2, marker="v")
		x = np.linspace(-2,10,100)
		y = (-cx/cy)*x + (-cb/cy)		
		axis.plot(x,y, 'r')

		return True












