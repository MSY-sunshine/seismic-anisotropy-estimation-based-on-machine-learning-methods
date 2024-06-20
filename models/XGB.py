import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time
import matplotlib.pyplot as plt


start = time.perf_counter()

# Load data
data = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/correct_direct_reflect_fre.csv")
y = data.loc[:, 'epsilon1']
X = data.loc[:, :'D_Trough_49']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

# Define the model
xgbr = xgb.XGBRegressor(
    gamma=0,
    min_child_weight=1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='exact',
    learning_rate=0.1,
    n_estimators=200,
    nthread=4,
    scale_pos_weight=1,
    seed=27,
    verbosity=0
)

# Evaluate the model
evalset = [(X_train, y_train), (X_test, y_test)]
xgbr.fit(X_train, y_train, eval_metric='logloss', eval_set=evalset)


# Make predictions using iteration_range
train_predict = xgbr.predict(X_train, iteration_range=(0, 200))
test_predict = xgbr.predict(X_test, iteration_range=(0, 200))

# Calculate RMSE and MAE for training set
rmse_train = np.sqrt(mean_squared_error(y_train, train_predict))
mae_train = mean_absolute_error(y_train, train_predict)
score_1 = r2_score(y_train, train_predict)
print("Training RMSE:", rmse_train)
print("Training MAE:", mae_train)
print("Training R2 Score:", score_1)

# Calculate RMSE and MAE for testing set
rmse_test = np.sqrt(mean_squared_error(y_test, test_predict))
mae_test = mean_absolute_error(y_test, test_predict)
score_2 = r2_score(y_test, test_predict)
print("Testing RMSE:", rmse_test)
print("Testing MAE:", mae_test)
print("Testing R2 Score:", score_2)

end = time.perf_counter()
print('Time elapsed: %s s' % (end - start))

# Feature importance using SHAP
import shap
shap.initjs()
explainer = shap.Explainer(xgbr)
shap_values = explainer(X_test)
plt.title('Delta(Nahr-Umer)')
shap.summary_plot(shap_values,X_test,plot_type='bar')


data2 = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/real_total.csv")

X_test2 = data2.loc[:,:'Fre_49']
ypred2 = xgbr.predict(X_test2,iteration_range=(0,200))
np.savetxt("result_xgb_target.txt",ypred2)