import numpy as np
import pandas as pd

class HousingDataset():
	
	def __init__(self):
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
		Y = self.binarize_feature(Y)
		self.X_train, self.Y_train = X[:304,:], Y[:304]
		self.X_test, self.Y_test = X[304:,:], Y[304:]
		
		
			
	def binarize_feature(self, feature):
		max_val = np.max(feature)
		threshold = max_val/2
		feature = np.where(feature<threshold, -1, 1)
			
		return feature