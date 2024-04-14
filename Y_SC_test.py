import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_list=pd.read_excel(r'D:\data.xlsx', header=0)
X=data_list[['urea','Cl','HCO','NH', 'temp', 'PDS']]
y=data_list['kobs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

#Enter test model parameters
reg =XGBRegressor()
reg = SVR()
reg = RandomForestRegressor()

y_scrambling_mses, y_scrambling_rmses, y_scrambling_maes, y_scrambling_r2s, y_scrambling_vars = [], [], [], [], []

results = pd.DataFrame(columns=['Iteration', 'TEST_MSE','TRAIN_MSE', 'TEST_RMSE', 'TRAIN_RMSE','TEST_MAE', 'TRAIN_MAE', 'TEST_R2', 'TRAIN_R2'])

for i in range(100):

    y_scrambled = np.random.permutation(y_train)
    reg.fit(X_train, y_scrambled)
    
    y_test_pred =reg.predict(X_test)
    y_train_pred =reg.predict(X_train)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    

    results = results._append({'Iteration': i+1, 
                               'TRAIN_MSE': train_mse, 
                               'TRAIN_RMSE': train_rmse, 
                               'TRAIN_MAE': train_mae, 
                               'TRAIN_R2': train_r2,
                               'TEST_MSE': test_mse, 
                               'TEST_RMSE': test_rmse, 
                               'TEST_MAE': test_mae, 
                               'TEST_R2': test_r2}, ignore_index=True)

results.to_excel('D:\\y_scrambling_results.xlsx', index=False)