from logistic_regressor import LogisticRegressor
#import numpy as np
#from iris_dataset import IrisDataset
from housing_dataset import HousingDataset
#import pandas as pd


def main():
	ds = HousingDataset()
	lg = LogisticRegressor(lr=0.5)
	lg.fit(ds.X_train, ds.Y_train, 10)
	Y_pred = lg.predict(ds.X_test)
	print(lg.accuracy(Y_pred, ds.Y_test))
    
if __name__ == "__main__":
    main()
	

	