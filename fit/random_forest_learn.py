from base import TreeQualityPredictor
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from rf_tqp import RFTreeQualityPredictor

rf = RandomForestRegressor(n_estimators=200)
tqp = RFTreeQualityPredictor(rf)

train_all = pd.read_csv("15_train.txt", sep = '\t', index_col = 0)
X = train_all.iloc[:,:-3]
y = train_all.iloc[:,-2]

tqp.fit(X, y)
tqp.save("rf_15_200.txt")