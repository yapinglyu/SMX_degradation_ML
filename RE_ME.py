import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 读取文件原始数据
data_list=pd.read_excel(r'data.xlsx', header=0)
X_nonstandard=data_list[['urea','Cl','HCO','NH','temp', 'PDS']]
y=data_list['kobs']

mse_scores = []
mae_scores = []
r2_scores = []
rmse_scores = []


num_predictions = 50

for i in range(num_predictions):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    cv_split = ShuffleSplit(n_splits=5, random_state=40)

    # Enter test model parameters
    reg = XGBRegressor()
    reg = SVR()
    reg = RandomForestRegressor()

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    rmse = np.sqrt(mse)
    rmse_scores.append(rmse)

scores_df = pd.DataFrame({
    'MSE': mse_scores,
    'MAE': mae_scores,
    'R2': r2_scores,
    'RMSE': rmse_scores
})

# 保存 DataFrame 为 Excel 文件
scores_df.to_excel('D:repeatability_measure.xlsx', index=False)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_predictions+1), mse_scores, label='MSE')
plt.plot(range(1, num_predictions+1), mae_scores, label='MAE')
plt.plot(range(1, num_predictions+1), r2_scores, label='R2')
plt.plot(range(1, num_predictions+1), rmse_scores, label='RMSE')
plt.xlabel('Prediction number')
plt.ylabel('Scores')
plt.legend()
plt.title('Repeatability Plot for Model')
plt.grid(True)
plt.show()

