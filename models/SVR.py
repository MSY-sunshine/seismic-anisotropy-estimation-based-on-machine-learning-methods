import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import time


start = time.perf_counter()

# Load data
data = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/correct_direct_reflect_fre.csv")
y = data.loc[:, 'epsilon1']
X = data.loc[:,:'D_Trough_49']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

# Initialize the scalers
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = x_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

# Transform the test data using the same scaler
X_test_scaled = x_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Now, you can train your model using the scaled data
# For example, using SVR
from sklearn.svm import SVR
model = SVR(C=1000, kernel='rbf')
model.fit(X_train_scaled, y_train_scaled.ravel())

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse_train = mean_squared_error(y_train_scaled, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train_scaled, y_train_pred)
r2_train = r2_score(y_train_scaled, y_train_pred)

mse_test = mean_squared_error(y_test_scaled, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_scaled, y_test_pred)
r2_test = r2_score(y_test_scaled, y_test_pred)

print(f"Training RMSE: {rmse_train}, MAE: {mae_train}, R2: {r2_train}")
print(f"Testing RMSE: {rmse_test}, MAE: {mae_test}, R2: {r2_test}")
end = time.perf_counter()
print('Time elapsed: %s s' % (end - start))


# Feature importance using SHAP
features = ['R_Peak_22','R_Trough_22','R_Peak_24','R_Trough_24','R_Peak_25','R_Trough_25','R_Peak_26','R_Trough_26','R_Peak_27','R_Trough_27','R_Peak_28','R_Trough_28','R_Peak_29','R_Trough_29','R_Peak_30','R_Trough_30','R_Peak_31','R_Trough_31','R_Peak_32','R_Trough_32','R_Peak_33','R_Trough_33','D_Peak_33','D_Trough_33','D_Peak_34','D_Trough_34','D_Peak_35','D_Trough_35','D_Peak_36','D_Trough_36','D_Peak_37','D_Trough_37','D_Peak_38','D_Trough_38','D_Peak_39','D_Trough_39','D_Peak_41','D_Trough_41','D_Peak_42',	'D_Trough_42',	'D_Peak_43','D_Trough_43','D_Peak_44',	'D_Trough_44','D_Peak_45','D_Trough_45','D_Peak_46','D_Trough_46','D_Peak_47','D_Trough_47','D_Peak_48','D_Trough_48','D_Peak_49',	'D_Trough_49','Fre_22','Fre_24','Fre_25','Fre_26','Fre_27','Fre_28','Fre_29','Fre_30','Fre_31','Fre_32','Fre_33','Fre_34','Fre_35','Fre_36','Fre_37','Fre_38','Fre_39','Fre_41','Fre_42','Fre_43','Fre_44','Fre_45','Fre_46','Fre_47','Fre_48','Fre_49']
import shap
shap.initjs()
explainer = shap.KernelExplainer(model.predict,X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values,X_test,plot_type='bar',feature_names=features)


# Predict on new data
data2 = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/real_total.csv")
X_test2 = data2.loc[:, :'Fre_49']
X_test2 = x_scaler.transform(X_test2)  # Use transform here
ypred2 = model.predict(X_test2)
ypred2 = y_scaler.inverse_transform(ypred2.reshape(-1, 1))
ypred2 = ypred2.ravel()

np.savetxt("result_MLP_target.txt", ypred2)