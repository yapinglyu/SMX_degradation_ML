import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

start_time = time.time()
data_list=pd.read_excel(r'D:\data.xlsx', header=0)
X=data_list[['urea','Cl','HCO','NH', 'temp', 'PDS']]
y=data_list['kobs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=65)
cv_split = ShuffleSplit(n_splits=5, random_state=40)

param_grid = dict(
    n_estimators=[i for i in range(1,1000,1)],
    max_depth=[i for i in range(1,20,1)],
    max_features=[i for i in range(1,7,1)],
    min_samples_leaf=[i for i in range(1,20,1)],
    min_samples_split=[i for i in range(1,20,1)],)

def count_space(param):
    no_option =1
    for i in param_grid:
        no_option*=len(param_grid[i])
    print(no_option)
count_space(param_grid)

rfc=RandomForestRegressor(random_state=90)
model= RandomizedSearchCV(estimator=rfc,
                            param_distributions=param_grid,
                            n_iter=200,
                            scoring='neg_mean_squared_error',
                            cv=cv_split,
                            random_state=90)
model.fit(X_train,y_train)
best_model=model.best_estimator_
print('best model：', model.best_params_)
y_test_predict = best_model.predict(X_test)
y_train_predict = best_model.predict(X_train)
print('prediction_y', y_test_predict )
print('trainMSE:', mean_squared_error(y_train, y_train_predict))
print('trainMAE',mean_absolute_error(y_train, y_train_predict))
print('trainRMSE:', np.sqrt(mean_squared_error(y_train, y_train_predict)))
print('train-r2', r2_score(y_train, y_train_predict))
print('testMSE:', mean_squared_error(y_test, y_test_predict))
print('testRMSE:', np.sqrt(mean_squared_error(y_test, y_test_predict)))
print('testMAE',mean_absolute_error(y_test, y_test_predict))
print('test-r2', r2_score(y_test, y_test_predict))
print(best_model.feature_importances_)

end_time = time.time()
run_time = end_time - start_time
print('Run time:', run_time)

