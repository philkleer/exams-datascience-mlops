# Regression Models with Scikit-Learn
# The objective of this test is to predict, using Regression models, the price of a house as a function of relevant variables representing different architectural, geographical and neighbourhood characteristics.

# The dataset used contains a large number of variables describing 1,460 homes according to a number of characteristics, as well as a target variable SalePrice containing the sale price of the property in question.

# The dataset is to be read from the house_price.csv file.

# (a) Run the following cell to import the packages required for the rest of the exercise.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from tqdm.notebook import *

import warnings

warnings.filterwarnings("ignore")

# (b) Read the house_price.csv in a Dataframe called hp. Pay attention to the index column.
hp = pd.read_csv("house_price.csv", index_col="Id")

hp.head(20)

hp.shape

# (c) Display the hp informations.
hp.info()

# some variables have a lot of missing information: Alley (only 91 non-null), FirePlaceQu (only 770 non-null), PoolQC (7 non-null),
# Fence (281 non-null) and MiscFeature (54 non-null)
# Probably need to get rid of these

# (d) Display the missing values in the dataset. Delete variables containing more than 80% missing data.
# Getting n per column
missings = hp.isna().sum()

print(missings)

# Getting percentage per column
missing_perc = hp.isna().mean() * 100

print(missing_perc[missing_perc > 80])

hp = hp.loc[:, missing_perc <= 80]

hp.head()

# (e) For each numerical variable, replace the missing values with the mean of the variable.
hp = hp.apply(
    lambda col: col.fillna(col.mean())
    if col.dtype == "int64" or col.dtype == "float64"
    else col
)

# (f) Transform each categorical variable into binary variables.
# it transforms the categorical variables of the dataset into dummies, keeping all
# Might it be better to take the argument drop_first=True to eliminate one of the types due to multicoliinearity in the regression model?
hp = pd.get_dummies(hp)

hp.head()

# (g) Create an object y containing the target variable SalePrice and an object X containing the rest of the variables.
y = hp["SalePrice"]
X = hp.drop(columns="SalePrice", axis=1)

# (h) Separate the data into a training set (X_train, y_train) and a test set (X_test, y_test), with 30% of the original data for the test set. Add random_state=42 to the train_test_split function to ensure reproducible results.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# (i) Define X_train_scaled and X_test_scaled by standardizing X_train and X_test using a well-known scikit-learn object.
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), index=X_test.index)

# (j) Create a linear regression model. Fit the model on the standardized train data. View model performance on standardized train and test data.
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

print("R2 Train data: ", lr.score(X_train_scaled, y_train))
print("R2 Test data: ", lr.score(X_test_scaled, y_test))

# (k) Calculate the root of the mean square error of this model on the standardized train and test data. Interpret the result.
y_pred_train = lr.predict(X_train_scaled)
y_pred_test = lr.predict(X_test_scaled)

print("MSE Train data: ", mean_squared_error(y_train, y_pred_train))
print("MSE Test data: ", mean_squared_error(y_test, y_pred_test))

# We see that the MSE of the test data is way higher than of the train data, indicating overfitting by the train data.

# The linear regression result is not good for the test set.

# We will try to improve it by using penalized models.

# We will start with a Ridge Regression. We use again the standardized data.

# (l) Fill in the blanks below to find the alpha that optimises the result. This code is used to do a manual GridSearchCV, and tqdm_notebook(alpha) is used to display the loop progress bar.
df_resultat_ridge = []
alphas = [0.01, 0.05, 0.1, 0.3, 0.8, 1, 5, 10, 15, 20, 35, 50]

for alpha in tqdm_notebook(alphas):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    mse_result = mean_squared_error(y_test, y_pred)
    res = pd.DataFrame({"Features": X_train.columns, "Coefficients": ridge.coef_})
    res["alpha"] = alpha
    res["mse"] = mse_result

    df_resultat_ridge.append(res)

df_resultat_ridge = pd.concat(df_resultat_ridge)

alphas_result = df_resultat_ridge.groupby("alpha")["mse"].mean()

alphas_result

# (m) Graph the MSE of the Ridge regression for each value of alphas.
plt.figure(figsize=[15, 10])

plt.plot(alphas_result.index, alphas_result.values, marker="x")
plt.ylabel("MSE")
plt.xlabel("Alpha level")
plt.title("MSE values depending of alpha for ridge regression")

# (n) From the graph, create a high-performance Ridge regression model. Fit the model on the standardized train data. View model performance on standardized train and test data.
# for me inspection of the graph it should be alpha = 35 since values are decreasing until 35 and afterwards slightly (!) increasing

ridge2 = Ridge(alpha=35)

ridge2.fit(X_train_scaled, y_train)

print("R2 Train data: ", ridge2.score(X_train_scaled, y_train))
print("R2 Test data: ", ridge2.score(X_test_scaled, y_test))

# Way better result!

# (o) Calculate the root of the mean square error of this model on the standardized train and test data. Interpret the result.
y_pred_train_ridge = ridge2.predict(X_train_scaled)
y_pred_test_ridge = ridge2.predict(X_test_scaled)

print("MSE Train data: ", mean_squared_error(y_train, y_pred_train_ridge))
print("MSE Test data: ", mean_squared_error(y_test, y_pred_test_ridge))

# Still the MSE is higher for test data, but the difference is way lower than before, indicating still overfitting
# however, it is not that severe as before

# Ridge regression made it possible to reduce overfitting.

# Let's see if we can further improve the score by using a model with a L1 and L2 norm penalty. We use standardized data again.

# (p) Apply a GridSearchCV for the model allowing a L1 L2 norm penalty. We will use 3-block cross-validation (i.e. with cv=3). The target metric will correspond to a minimization of the mean_squared_error. You will find the list of scoring methods here if necessary. We will choose as parameters:
alpha = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

l1_ratio = np.arange(0.0, 1.01, 0.05)

elastic_net = ElasticNet()

params = {
    "alpha": [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "l1_ratio": np.arange(0.0, 1.01, 0.05),
}

grid_search = GridSearchCV(
    estimator=elastic_net,
    param_grid=params,
    cv=3,
    #    scoring='neg_mean_squared_error',
    n_jobs=-1,
)

grid_search.fit(X_train_scaled, y_train)

# Showing the best solution
print("Best alpha: ", grid_search.best_params_["alpha"])
print("Best L1 ratio: ", grid_search.best_params_["l1_ratio"])

# (q) Fill in the code to display the score for each combination of alpha and l1_ratio. To do this, use the GridSearchCV documentation and look for the attributes that answer the question.
alphas = grid_search.cv_results_["param_alpha"]
l1_ratios = grid_search.cv_results_["param_l1_ratio"]
scores = grid_search.cv_results_["mean_test_score"]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

colors = scores

scatter = ax.scatter(alphas, l1_ratios, scores, c=colors, cmap="viridis")
fig.colorbar(scatter)
ax.set_xlabel("Alpha")
ax.set_ylabel("L1 Ratio")
ax.set_zlabel("Score")
ax.set_title("Results of the GridSearch")

plt.show()

# (r) Create the L1 L2 penalty model with the best parameters obtained by the GridSearchCV. Fit the model on the standardized train data. View model performance on standardized train and test data.
elastic_net = ElasticNet(
    alpha=grid_search.best_params_["alpha"],  # 1.0,
    l1_ratio=grid_search.best_params_["l1_ratio"],  # 0.35000000000000003
)

elastic_net.fit(X_train_scaled, y_train)


print("R2 Train data: ", elastic_net.score(X_train_scaled, y_train))
print("R2 Test data: ", elastic_net.score(X_test_scaled, y_test))

# (s) Calculate the root of the mean square error of this model on the standardized train and test data. Interpret the result.
y_pred_train_en = elastic_net.predict(X_train_scaled)
y_pred_test_en = elastic_net.predict(X_test_scaled)

print("MSE Train data: ", mean_squared_error(y_train, y_pred_train_en))
print("MSE Test data: ", mean_squared_error(y_test, y_pred_test_en))

# Still we see that the train data mse is lower than the test data mse, indicating some kind of overfitting in
# the training data. The ridge results (optimized model ridge2) indicate lower MSE, therefore, Ridge models have better model for teh data
# than the elastic net.
