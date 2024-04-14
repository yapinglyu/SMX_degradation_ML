import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ShuffleSplit
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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

param_grid = dict(
    C=[i for i in range(1,500,1)],
    gamma=[i/100 for i in range(1,50,1)],
    kernel=["rbf"])

def count_space(param):
    no_option = 1
    for i in param_grid:
        no_option *= len(param_grid[i])
    print(no_option)
count_space(param_grid)

rbf_svr = SVR()
model= RandomizedSearchCV(estimator=rbf_svr,
                            param_distributions=param_grid,
                            n_iter=100,
                            scoring='neg_mean_squared_error',
                            cv=cv_split,
                            random_state=90)
model.fit(X_train,y_train)
best_model=model.best_estimator_
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