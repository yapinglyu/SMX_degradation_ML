import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from time import time

time0=time()
data_list=pd.read_excel(r'D:\data.xlsx', header=0)
data_list.head()
X=data_list.iloc[:,:-1]
y =data_list.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

cv_split = ShuffleSplit(n_splits=5, random_state=40)
reg = XGBR(random_state=40)
params = dict(
    n_estimators=[i for i in range(0,1000,1)],
    max_depth=[i for i in range(1,20,1)], 
    min_child_weight=[i for i in range(0,10,1)], 
    gamma=[i/10.0 for i in range(0,10,1)],
    subsample=[i/10.0 for i in range(1,9,1)],
    colsample_bytree=[i/10.0  for i in range(0,10,1)],
    learning_rate=[i/100.0 for i in range(0,10,1)], )

def count_space(param):
    no_option =1
    for i in params:
        no_option*=len(params[i])
    print(no_option)
count_space(params)

model= RandomizedSearchCV(estimator=reg,
                            param_distributions=params,
                            n_iter=200,
                            scoring='neg_mean_squared_error',
                            cv=cv_split,
                            random_state=40)
model.fit(X_train, y_train)
best_model=model.best_estimator_
best_score=model.best_score_
y_test_predict = best_model.predict(X_test)
y_train_predict = best_model.predict(X_train)
print('pridiction_y：', y_test_predict)
print('trainMSE:', mean_squared_error(y_train, y_train_predict))
print('trainRMSE:', np.sqrt(mean_squared_error(y_train, y_train_predict)))
print('testMSE:', mean_squared_error(y_test, y_test_predict))
print('testRMSE:', np.sqrt(mean_squared_error(y_test, y_test_predict)))
print('train-r2', r2_score(y_train, y_train_predict))
print('test-r2', r2_score(y_test, y_test_predict))
print('time:',time()-time0)





