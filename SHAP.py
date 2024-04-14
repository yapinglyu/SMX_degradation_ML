import numpy as np
import pandas as pd
import graphviz
import shap
from xgboost import xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_list=pd.read_excel(r'D:\data.xlsx', header=0)
X=data_list[['urea','Cl','HCO','NH', 'temp', 'PDS']]
y=data_list['kobs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

#Enter test model parameters
reg =xgb()
reg = SVR()
reg = RandomForestRegressor()


digraph =xgb.to_graphviz(reg, num_trees=0)
digraph.format = 'png'
digraph.view('./iris_xgb')
digraph2 =xgb.to_graphviz(reg, num_trees=549)
digraph2.format = 'png'
digraph2.view('./2iris_xgb')
explainer=shap.TreeExplainer(reg)
shap_values=explainer(X)

shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)
shap_interaction_values = shap.TreeExplainer(reg).shap_interaction_values(X)
shap.summary_plot(shap_interaction_values, X, max_display=6)

shap.dependence_plot("Cl", shap_values.values, X)
shap.dependence_plot("HCO", shap_values.values, X)
shap.dependence_plot("temp", shap_values.values, X)
shap.dependence_plot("urea", shap_values.values, X)
shap.dependence_plot("NH", shap_values.values, X)
