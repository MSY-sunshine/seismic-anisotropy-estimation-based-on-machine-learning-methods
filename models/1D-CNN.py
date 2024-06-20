import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Start timer
start = time.perf_counter()

# Load data
data = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/correct_direct_reflect_fre.csv")
y = data.loc[:, 'epsilon1'].values.reshape(-1, 1)  # Reshape y to be a 2D array
X = data.loc[:, :'D_Trough_49']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

# Initialize the scalers
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# Fit the scalers on the training data and transform it
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# Transform X_train and X_test to match the expected input shape for 1D-CNN
X_train = X_train[:, :, np.newaxis]
X_train = X_train.transpose(0, 2, 1)
X_train = torch.tensor(X_train).float()

X_test = X_test[:, :, np.newaxis]
X_test = X_test.transpose(0, 2, 1)
X_test = torch.tensor(X_test).float()

# Convert y_train and y_test to torch tensors
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

# Create datasets and data loaders
datasets = torch.utils.data.TensorDataset(X_train, y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size=50, shuffle=True)


# Define the model
class OneD_CNN(nn.Module):
    def __init__(self):
        super(OneD_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        self.fc1 = nn.Linear(52, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = OneD_CNN()
model.train()
criterion = nn.MSELoss(reduction='sum')

# Training loop
num_epochs = 1000
train_loss = []

for epoch in range(num_epochs):
    running_loss = 0.0
    lr = 0.01 * (0.9 ** (min(epoch - 1, 200) // 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    for inputs, labels in train_iter:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    if epoch % 10 == 0:
        print(
            'Epoch[%d]/[%d] running accumulative loss across all batches: %.3f' % (epoch + 1, num_epochs, running_loss))

# Evaluation
model.eval()
with torch.no_grad():
    train_predict = model(X_train)
    test_predict = model(X_test)

# Calculate metrics
train_predict_np = train_predict.detach().numpy()
test_predict_np = test_predict.detach().numpy()
y_train_np = y_train.detach().numpy()
y_test_np = y_test.detach().numpy()

# R2 Score
r2_train = r2_score(y_train_np, train_predict_np)
r2_test = r2_score(y_test_np, test_predict_np)
print("Training R2 Score:", r2_train)
print("Testing R2 Score:", r2_test)

# Mean Squared Error
mse_train = mean_squared_error(y_train_np, train_predict_np)
mse_test = mean_squared_error(y_test_np, test_predict_np)

# Root Mean Squared Error
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
print("Training RMSE:", rmse_train)
print("Testing RMSE:", rmse_test)

# Mean Absolute Error
mae_train = mean_absolute_error(y_train_np, train_predict_np)
mae_test = mean_absolute_error(y_test_np, test_predict_np)
print("Training MAE:", mae_train)
print("Testing MAE:", mae_test)

# End timer
end = time.perf_counter()
print('Time elapsed: %s s' % (end - start))

# Feature importance using SHAP
features = ['R_Peak_22','R_Trough_22','R_Peak_24','R_Trough_24','R_Peak_25','R_Trough_25','R_Peak_26','R_Trough_26','R_Peak_27','R_Trough_27','R_Peak_28','R_Trough_28','R_Peak_29','R_Trough_29','R_Peak_30','R_Trough_30','R_Peak_31','R_Trough_31','R_Peak_32','R_Trough_32','R_Peak_33','R_Trough_33','D_Peak_33','D_Trough_33','D_Peak_34','D_Trough_34','D_Peak_35','D_Trough_35','D_Peak_36','D_Trough_36','D_Peak_37','D_Trough_37','D_Peak_38','D_Trough_38','D_Peak_39','D_Trough_39','D_Peak_41','D_Trough_41','D_Peak_42',	'D_Trough_42',	'D_Peak_43','D_Trough_43','D_Peak_44',	'D_Trough_44','D_Peak_45','D_Trough_45','D_Peak_46','D_Trough_46','D_Peak_47','D_Trough_47','D_Peak_48','D_Trough_48','D_Peak_49',	'D_Trough_49','Fre_22','Fre_24','Fre_25','Fre_26','Fre_27','Fre_28','Fre_29','Fre_30','Fre_31','Fre_32','Fre_33','Fre_34','Fre_35','Fre_36','Fre_37','Fre_38','Fre_39','Fre_41','Fre_42','Fre_43','Fre_44','Fre_45','Fre_46','Fre_47','Fre_48','Fre_49']
import shap
shap.initjs()
explainer = shap.DeepExplainer(model,X_train)
shap_values = explainer.shap_values(X_test)
plt.title('Epsilon(Nahr-Umer)')
shap.summary_plot(shap_values[0],plot_type='bar',feature_names=features)

# Predict on new data
data2 = pd.read_csv("/Users/zhaoguibin/Desktop/real_data_offset/real_total.csv")
X_test2 = data2.loc[:, :'Fre_49']
X_test2 = x_scaler.transform(X_test2)
X_test2 = X_test2[:, :, np.newaxis]
X_test2 = X_test2.transpose(0, 2, 1)
X_test2 = torch.tensor(X_test2).float()
ypred2 = model(X_test2)

np.savetxt("result_1DCnn_target.txt", ypred2.detach().numpy())
