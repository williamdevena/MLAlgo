import numpy as np
import pandas as pd

class HousingDataset():
	
	def __init__(self, num_labels):
		self.num_labels = num_labels
		csv_url = 'https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt'
		namecol = [    'CRIM',
			                   'ZN',
			                'INDUS', 
			                 'CHAS',
			                  'NOX',
			                   'RM',
			                  'AGE',
			                  'DIS',
			                  'RAD',
			                  'TAX',
			              'PTRATIO',
			                    'B',
			                'LSTAT',
			                 'MEDV']

		ds = (pd.read_csv(csv_url, names=namecol, sep='\s+')).sample(frac=1)
		X = ds.iloc[:,:13].values
		Y = ds.iloc[:,13].values

		Y = self.label_feature(self.num_labels, Y)
			
		self.X_train, self.Y_train = X[:304,:], Y[:304]
		self.X_test, self.Y_test = X[304:,:], Y[304:]
		
				
	def binarize_feature(self, feature):
		max_val = np.max(feature)
		threshold = max_val/2
		feature = np.where(feature<threshold, -1, 1)
			
		return feature
	
	
	def label_feature(self, num_labels, feature):
		max_val = np.max(feature)
		step = int(max_val/num_labels)	
		for x in range(num_labels):
			#if max_val%step!=0 and x==(num_labels-1):
			if x==(num_labels-1):
				feature = np.where(feature>=(step*x), x, feature)
			else:
				feature=np.where(((feature<(step*(x+1))) & (feature>=(step*x))), x, feature)
		
		#feature = np.where(feature<threshold, -1, 1)
			
		return feature
	
	
def main():
	ds = HousingDataset()
	#print(type(ds.Y_test))
	#print(ds.Y_test)
	#print(ds.label_feature(3, ds.Y_test))
	

	
if __name__=="__main__":
	main()
	
