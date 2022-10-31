from logistic_regressor import LogisticRegressor
#import numpy as np
from iris_dataset import IrisDataset
from housing_dataset import HousingDataset
#import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split




def main():
	data = load_breast_cancer()
	target = data.target[[10, 50, 85]]
	
	data['target'] = np.where(data['target']==0, -1, 1)	
	
	X_train, X_test = train_test_split(data['data'], test_size=0.2)
	Y_train, Y_test = train_test_split(data['target'], test_size=0.2)
	
	lg = LogisticRegressor(lr=0.5)
	lg.fit2(X_train, Y_train, 100)
	Y_pred = lg.predict(X_test)
	print(lg.accuracy(Y_pred, Y_test))
 	
    
if __name__ == "__main__":
    main()
	
	

	