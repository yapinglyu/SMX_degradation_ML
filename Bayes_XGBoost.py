import numpy as np 
import pandas as pd 
from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from bayes_opt import BayesianOptimization
from time import time

time0= time()
data_list=pd.read_excel(r'D:\data.xlsx', header=0)
X=data_list[['urea','Cl','HCO','NH', 'temp', 'PDS']]
y=data_list['kobs']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
cv_split = ShuffleSplit(n_splits=5, random_state=40)

def _xgb_evaluate(n_estimators, learning_rate, max_depth, subsample, gamma, colsample_bytree, min_child_weight,alpha): #定义所有想要调节的参数 n_estimators, learning_rate, max_depth, subsample, gamma, colsample_bytree, min_child_weight,
    reg = XGBR (n_estimators= int(n_estimators), 
              learning_rate=learning_rate, 
              max_depth=int(max_depth),
              subsample=subsample,
              gamma= gamma,
              colsample_bytree=colsample_bytree,
              min_child_weight=min_child_weight,
              alpha=alpha,
              random_state=40)
    
    cv_result = cross_val_score(reg, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv_split, error_score='raise')
    return np.mean(cv_result)

xgb_bo = BayesianOptimization(_xgb_evaluate, {'n_estimators': (0,1000), 
                                              'learning_rate': (0,0.1),
                                              'max_depth': (1, 40), 
                                              'subsample': (0,1), 
                                              'gamma': (0,0.9), 
                                              'colsample_bytree': (0, 1), 
                                              'min_child_weight': (0, 1),
                                              'alpha':(0, 2) 
                                              }, random_state=40)

xgb_bo.maximize(init_points=10, n_iter=500) 
params_best =xgb_bo.max["params"]
score_best = xgb_bo.max["target"]

print('best params', params_best, '\n', 'best score', score_best)

reg_val = XGBR(n_estimators= int(params_best["n_estimators"]), #默认参数输入一定是浮点数，因此需要套上int函数处理成整数
              learning_rate=params_best["learning_rate"], #需要调整的超参数等于目标函数的输入，不需要调整的超参数直接等于固定值
              max_depth=int(params_best["max_depth"]),
              subsample=params_best["subsample"],
              gamma= params_best["gamma"],
              colsample_bytree=params_best["colsample_bytree"],
              min_child_weight=params_best["min_child_weight"],
              random_state=40)

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
