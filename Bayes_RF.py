import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from bayes_opt import BayesianOptimization
from time import time

time0= time()
data_list=pd.read_excel(r'D:\data.xlsx', header=0)
X=data_list[['urea','Cl','HCO','NH', 'temp', 'PDS']]
y=data_list['kobs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=65)

cv_split = ShuffleSplit(n_splits=5, train_size=0.7, test_size=0.3, random_state=40)

def _rf_evaluate(n_estimators, max_features, max_depth,min_samples_leaf,min_samples_split ): 
    reg = RandomForestRegressor(n_estimators= int(n_estimators), 
                                max_features= int(max_features),
                                max_depth=int(max_depth),
                                min_samples_leaf=int(min_samples_leaf),
                                min_samples_split=int(min_samples_split),
                                random_state=90)

    cv_result = cross_val_score(reg, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv_split, error_score='raise')
    return np.mean(cv_result)

rf_bo = BayesianOptimization(_rf_evaluate, {'n_estimators': (1,1000),
                                            'max_features': (1,6),
                                            'max_depth': (1, 30),
                                            'min_samples_leaf': (1,10),
                                            'min_samples_split': (1,10),
                                            }, random_state=90)

rf_bo.maximize(init_points=10, n_iter=100)
params_best =rf_bo.max["params"]
score_best = rf_bo.max["target"]

print('best params', params_best, '\n', 'best score', score_best)

reg_val = RandomForestRegressor(n_estimators= int(params_best["n_estimators"]), #默认参数输入一定是浮点数，因此需要套上int函数处理成整数
                                max_features=int(params_best["max_features"]),
                                max_depth=int(params_best["max_depth"]),
                                min_samples_leaf=int(params_best["min_samples_leaf"]),
                                min_samples_split=int(params_best["min_samples_split"]),
                                random_state=90)
best_model=reg_val.fit(X_train,y_train)
print(best_model)
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
