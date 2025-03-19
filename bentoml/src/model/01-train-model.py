import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import bentoml
import pickle

print('Loading data ...')

# loading data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# scaling
scaler = StandardScaler()   
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# saving scaled data
X_train_scaled.to_csv('data/processed/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test_scaled.csv', index=False)

print('Scaled data is saved in \'data/processed/\'')

print('Fitting training model ...')
# creating model
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

print('Predicting test model ...')
# test model
y_pred = linreg.predict(X_test_scaled)

# generating scores
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# printing
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Model fit is awesome

print('Registering model for bento (incl. scaler) ...')

# Save the model in BentoML's Model Store
# I added saving the scaler, since users will only enter raw data and we need to scale data before predictions
model_ref = bentoml.sklearn.save_model(
    "linreg_admission",
    linreg,
    custom_objects={"scaler": pickle.dumps(scaler)} 
)

print(f"Model saved as: {model_ref}")

print('All done!')
