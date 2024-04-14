import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr



data_list=pd.read_excel(r'D:\data.xlsx', header=0)
X=data_list[['urea','Cl','HCO','NH', 'temp', 'PDS']]
y=data_list['kobs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Enter test model parameters
reg = RandomForestRegressor()
reg =XGBRegressor()
reg = SVR()


n_permutations = 100  # 置换次数
Q2 = []
R2 = []
CO = []
for _ in range(n_permutations):

    y_permuted = np.random.permutation(y)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_permuted, y_test_permuted = y_permuted[train_index], y_permuted[test_index]

        reg.fit(X_train, y_train_permuted)
        y_pred_permuted =reg.predict(X_test)

        q2 = r2_score(y_test, y_pred_permuted)
        r2 = r2_score(y_train, y_train_permuted)
        co, _ = pearsonr(y, y_permuted)
        Q2.append(q2)
        R2.append(r2)
        CO.append(co)
        print('Q2:',Q2)
        print('R2:', R2)
        print('CO:', CO)


lr1 = LinearRegression()
lr2 = LinearRegression()
validation_q2_scores = np.array(Q2).reshape(-1, 1)
validation_r2_scores = np.array(R2).reshape(-1, 1)
correlation_coefficients = np.array(CO).reshape(-1, 1)

lr1.fit(correlation_coefficients,validation_q2_scores)
lr2.fit(correlation_coefficients,validation_r2_scores)
slope = lr1.coef_
intercept1 = lr1.intercept_
intercept2 = lr2.intercept_

print(f"Q2: {intercept1[0]:.4f}")
print(f"R2: {intercept2[0]:.4f}")


