# Machine Learning (Classification) with *scikit-learn*

# Our study focuses on the energy efficiency of residential buildings, and more particularly on heating and cooling needs, according to architectural features such as wall surface, glazed area, orientation, ...

# The dataset used contains eight attributes describing its features for 768 buildings and 2 target attributes: the heating loads and the cooling loads of these buildings.

# The purpose of the exercise is to predict the charges for each building, according to the first eight attributes.

# The dataset is to be read in the file "ENB_data.csv". Note that the columns are separated by ';'.

# Run the following cell to import the libraries needed for the exercise.
# Load the file "ENB_data.csv" and make a first exploration of the data in a data frame df.

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier

### Enter your code here
df = pd.read_csv("ENB_data.csv", sep=";")

df.info()
df.head(20)


# Analyze the correlations between all the variables in df.
# Which explanatory variables are most correlated to the two target variables ?
df.corr()

# mostly correlated to heating_load: roof_area (-0.86), overall_height (0.89), glazing_area_distribution (0.87)
# mostly correlated to cooling_load: roof_area (-0.89), overall_height (0.89), surface_area (-0.67)
# therefore for both together, it is roof_area and overall_height

# The next step is to create an optimal classification model after grouping buildings into classes according to the total energy loads (heating + cooling).

# Create a new column at df, called total_charges, totaling for each building the heating and cooling loads.
# In a new variable charges_classes, split the buildings into 4 distinct classes with labels 0, 1, 2, 3 according to the 3 quantiles of the new variable created.

df["total_charges"] = df["heating_load"] + df["cooling_load"]

df["charges_classes"] = pd.qcut(df["total_charges"], labels=[0, 1, 2, 3], q=4)

# Store the explanatory variables in a data table.
# Seperate the data into a training set and a test set (20 %). Make sure to specify charges_classes as the target variable.
# Use the scaled standardization transformation on the explanatory variables appropriately.
data = df.drop(columns="charges_classes", axis=1)
target = df["charges_classes"]

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=472
)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

# In what follows, we will compare several methods of learning. For each of them, it will be necessary to explore the interval of the next hyperparameters :

# K-nearest neighbors. Hyperparameter to adjust :
# 'n_neighbors' : 2 to 50.

# SVM. Hyperparameter to adjust :
# kernel : 'rbf', 'linear'.
# C : 0.1 ; 1 ; 10 ; 50 .

# RandomForest. Hyperparameter to adjust :
# 'max_features': "sqrt", "log2", None
# 'min_samples_split': Number of paires ranging from 2 to 30.

# For each algorithm mentioned above:

# Select the hyperparameters by cross-validation on the learning sample

# Display hyperparameters retained

# Apply the model to the test set, display the confusion matrix and the model score on the test set

# Which model has the best accuracy?

# K-nearest neighbors
from sklearn import neighbors

# Initializing classifier
knn = neighbors.KNeighborsClassifier()

# Setting parameter search
parameters = {"n_neighbors": list(range(2, 51))}

# run grid search
knn_cv = model_selection.GridSearchCV(knn, param_grid=parameters, cv=7)

# fit on scaled train data
knn_cv.fit(X_train_scaled, y_train)

# print best parameter
print("Best Parameter: ", knn_cv.best_params_)

# Scale Test data
X_test_scaled = scaler.transform(X_test)

# predict on test data
y_pred = knn_cv.predict(X_test_scaled)

print("Accuracy Score KNN: ", knn_cv.score(X_test_scaled, y_test))

# SVM
from sklearn import svm

# Initializing classifier
svm = svm.SVC()

# Setting parameter search
parameters = {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10, 50]}

# run grid search
svm_cv = model_selection.GridSearchCV(svm, param_grid=parameters, cv=7)

# fit on scaled train data
svm_cv.fit(X_train_scaled, y_train)

# print best parameter
print("Best Parameter: ", svm_cv.best_params_)

# Scale Test data
X_test_scaled = scaler.transform(X_test)

# predict on test data
y_pred = svm_cv.predict(X_test_scaled)

print("Accuracy Score SVM: ", svm_cv.score(X_test_scaled, y_test))

# Random Forest
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier

# Initializing classifier
rf = RandomForestClassifier()

# Setting parameter search
parameters = {
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": list(range(2, 30)),
}

# run grid search
rf_cv = model_selection.GridSearchCV(rf, param_grid=parameters, cv=7)

# fit on scaled train data
rf_cv.fit(X_train_scaled, y_train)

# print best parameter
print("Best Parameter: ", rf_cv.best_params_)

# Scale Test data
X_test_scaled = scaler.transform(X_test)

# predict on test data
y_pred = rf_cv.predict(X_test_scaled)

print("Accuracy Score Random Forest: ", rf_cv.score(X_test_scaled, y_test))

print("Overall comparison\n")

print("Accuracy Score KNN: ", round(knn_cv.score(X_test_scaled, y_test), 4), "\n")
print("Accuracy Score SVM: ", round(svm_cv.score(X_test_scaled, y_test), 4), "\n")
print(
    "Accuracy Score Random Forest: ", round(rf_cv.score(X_test_scaled, y_test), 4), "\n"
)

print("The overall best score is achieved by Random Forest with 0.9935")

# Voting classifier

# In this part, you will have to implement an ensemble method

# Create a VotingClassifier object that you will store in a variable named vc. Use the 3 previous models as arguments, and set vote to hard.
# Does this model give better results ?
from sklearn.ensemble import VotingClassifier
from sklearn import svm

# redefining objects as best model from above grid search
knn_best = neighbors.KNeighborsClassifier(n_neighbors=5)
svm_best = svm.SVC(C=50, kernel="linear")
rf_best = RandomForestClassifier(max_features="sqrt", min_samples_split=13)

# Initializing Classifier
vclf = VotingClassifier(
    estimators=[("knn", knn_best), ("SVM", svm_best), ("Random Forest", rf_best)],
    voting="hard",
)

# Training Classifier
vclf.fit(X_train_scaled, y_train)

# Testing classifier
vclf.score(X_test, y_test)

print("Accuracy VCLF: ", vclf.score(X_test, y_test))

# The model performs significantly worse.
