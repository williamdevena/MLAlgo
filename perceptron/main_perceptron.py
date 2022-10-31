from iris_dataset import IrisDataset
from perceptron2 import Perceptron

def main():
	epochs = 100
	ds = IrisDataset()
	p = Perceptron()
	print(p.fit(epochs, ds.points, ds.labels))

if __name__ == '__main__':
	main()

