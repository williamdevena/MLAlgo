import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class IrisDataset():
	def __init__(self, all_features, all_ds):
		self.num_features = all_features
		self.all_ds = all_ds
		
		ds = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
		
		ds = ds.iloc[:100,:].sample(frac=1)
		
		self.labels=ds.iloc[:,-1].values
		self.labels = np.where(self.labels == "Iris-setosa", -1, 1)
		self.points=ds.iloc[:, :4].values
		
		self.X_train, self.Y_train = self.points[:60], self.labels[:60]
		self.X_test, self.Y_test = self.points[60:], self.labels[60:]
			
		self.points = [np.insert(flower, 0, 1) for flower in self.points]
		

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

