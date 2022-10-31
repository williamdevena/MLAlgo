import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class IrisDataset():
	def __init__(self):
		ds = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data').sample(frac=1)
		y=ds.iloc[:,4].values
		self.labels = np.where(y == "Iris-setosa", -1, np.where(y == 'Iris-versicolor', 1, 0))
		x=ds.iloc[:, [0,1,2,3]].values
		self.points = [np.insert(flower, 0, 1) for flower in x]

	def visualize(self):
		x1 = [self.points[index][1] for index in range(len(self.points)) if self.labels[index]==-1]
		y1 = [self.points[index][2] for index in range(len(self.points)) if self.labels[index]==-1]
		x2 = [self.points[index][1] for index in range(len(self.points)) if self.labels[index]==1]
		y2 = [self.points[index][2] for index in range(len(self.points)) if self.labels[index]==1]
		plt.scatter(x1,y1, marker="o")
		plt.scatter(x2,y2, marker="v")
		plt.show()

def main():
	ds = IrisDataset()
	ds.visualize()



if __name__ == '__main__':
	main()

