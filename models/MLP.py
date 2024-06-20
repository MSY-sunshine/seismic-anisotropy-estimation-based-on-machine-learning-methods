import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import time
import matplotlib.pyplot as plt
import shap

# Start timer
start = time.perf_counter()

# Load data
data = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/correct_direct_reflect_fre.csv")
y = data.loc[:, 'delta1'].values.reshape(-1, 1)  # Reshape y to be a 2D array
X = data.loc[:, :'D_Trough_49']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

# Initialize the scalers
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# Fit the scalers on the training data and transform it
X_train = x_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)

# Transform the test data using the same scalers
X_test = x_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

# Train MLP model
MLP = MLPRegressor(hidden_layer_sizes=(64, 256, 64, 32, 16, 8), activation='relu', solver='adam', alpha=0.005, max_iter=8000)
print("Fitting model right now")
MLP.fit(X_train, y_train.ravel())

# Make predictions on training set
train_predict = MLP.predict(X_train)

# Calculate metrics for training set
mse_train = mean_squared_error(y_train, train_predict)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_predict)
r2_train = r2_score(y_train, train_predict)

print("Training RMSE:", rmse_train)
print("Training MAE:", mae_train)
print("Training R2 Score:", r2_train)

# Make predictions on testing set
test_predict = MLP.predict(X_test)

# Calculate metrics for testing set
mse_test = mean_squared_error(y_test, test_predict)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, test_predict)
r2_test = r2_score(y_test, test_predict)

print("Testing RMSE:", rmse_test)
print("Testing MAE:", mae_test)
print("Testing R2 Score:", r2_test)

# End timer
end = time.perf_counter()
print('Time elapsed: %s s' % (end - start))

# Feature importance using SHAP
features = ['R_Peak_22','R_Trough_22','R_Peak_24','R_Trough_24','R_Peak_25','R_Trough_25','R_Peak_26','R_Trough_26','R_Peak_27','R_Trough_27','R_Peak_28','R_Trough_28','R_Peak_29','R_Trough_29','R_Peak_30','R_Trough_30','R_Peak_31','R_Trough_31','R_Peak_32','R_Trough_32','R_Peak_33','R_Trough_33','D_Peak_33','D_Trough_33','D_Peak_34','D_Trough_34','D_Peak_35','D_Trough_35','D_Peak_36','D_Trough_36','D_Peak_37','D_Trough_37','D_Peak_38','D_Trough_38','D_Peak_39','D_Trough_39','D_Peak_41','D_Trough_41','D_Peak_42','D_Trough_42','D_Peak_43','D_Trough_43','D_Peak_44','D_Trough_44','D_Peak_45','D_Trough_45','D_Peak_46','D_Trough_46','D_Peak_47','D_Trough_47','D_Peak_48','D_Trough_48','D_Peak_49','D_Trough_49','Fre_22','Fre_24','Fre_25','Fre_26','Fre_27','Fre_28','Fre_29','Fre_30','Fre_31','Fre_32','Fre_33','Fre_34','Fre_35','Fre_36','Fre_37','Fre_38','Fre_39','Fre_41','Fre_42','Fre_43','Fre_44','Fre_45','Fre_46','Fre_47','Fre_48','Fre_49']
shap.initjs()
explainer = shap.KernelExplainer(MLP.predict, X_train)
shap_values = explainer.shap_values(X_test)
plt.title('Delta(Nahr-Umer)')
shap.summary_plot(shap_values, X_test, plot_type='bar', feature_names=features)

# Predict on new data
data2 = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/real_total.csv")
X_test2 = data2.loc[:, :'Fre_49']
X_test2 = x_scaler.transform(X_test2)  # Use transform here

ypred2 = MLP.predict(X_test2)
ypred2 = y_scaler.inverse_transform(ypred2.reshape(-1, 1))
ypred2 = ypred2.ravel()

np.savetxt("result_MLP_target.txt", ypred2)
