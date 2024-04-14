import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import time

start_time = time.time()
data_list=pd.read_excel(r'D:\data.xlsx', header=0)
X_nonstandard=data_list[['urea','Cl','HCO','NH','temp', 'PDS']]
y=data_list['kobs']

X_nonstandard = np.array(X_nonstandard)
Mm = MinMaxScaler()
X = Mm.fit_transform(X_nonstandard)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=147)

cv_split = ShuffleSplit(n_splits=5, random_state=40)

def _svr_evaluate(C, gamma):
    reg = SVR (C=C,
              gamma=gamma)
    cv_result = cross_val_score(reg, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv_split,
                                error_score='raise')
    return np.mean(cv_result)

svr_bo = BayesianOptimization(_svr_evaluate, {'C': (0,500),
                                              'gamma': (0,0.5),
                                              }, random_state=40)
svr_bo.maximize(init_points=10, n_iter=200)

params_best =svr_bo.max["params"]
score_best = svr_bo.max["target"]

print('best params', params_best, '\n', 'best score', score_best)
reg_val = SVR(C=params_best["C"],
              gamma=params_best["gamma"],
              kernel='rbf')

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

end_time = time.time()
run_time = end_time - start_time
print('Run time:', run_time)