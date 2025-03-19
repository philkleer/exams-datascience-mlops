# Anomaly Detection

# Context

# Anomaly detection is an issue that affects many sectors, including banking and IT. As a result, the detection of online banking fraud is essential to the development of e-commerce and online payments.

# In this exam, we focus on the detection of online banking fraud. To this end, we have access to the file fraud_data.csv, which contains data on online banking transactions. To be more precise, each observation represents an online transaction. Column variables have the following meaning :

# |Variable | Description| |----------|------------| |'step'| unit of time such that 1 step equals 1 hour| |'type'| type of the online transaction| |'amount'| amount of the online transaction| |'nameOrig'| name of the customer who carried out the online transaction| |'oldbalanceOrg'| customer's balance prior to the online transaction| |'newbalanceOrig'| customer's balance after the online transaction| |'nameDest'| name of the receiver of the online transaction| |'oldbalanceDest'| receiver's balance before the online transaction| |'newbalanceDest'| receiver balance after the online transaction| |'isFraud'| whether the online transaction is fraudulent (1) or not (0)|

# As the dataset is unbalanced with a majority of non-fraudulent online transactions, an online fraud can be considered an anomaly within the data.

# The goal of this exam is to predict whether an online transaction is fraudulent or not, using anomaly detection algorithms.

# (a) Load the packages you'll need later.

# (b) Import the file fraud_data.csv into a DataFrame called df, specifying that the first column contains the indexes.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
import seaborn as sns

df = pd.read_csv("fraud_data.csv", index_col=0)

df.head()

# (c) Explore the dataframe and check that this is an anomaly detection problem.
print(df.info())
print(df.describe())


print("Mean amount among frauds:", df["amount"][df["isFraud"] == 1].mean(), end="\n\n")
print(
    "Mean amount among non-frauds:", df["amount"][df["isFraud"] == 0].mean(), end="\n\n"
)

# We see a huge difference among transactions marked as fraud, underlying

plt.boxplot([df.amount, df.isFraud])

df["amount"].value_counts()

# (d) Do the necessary pre-processing to obtain a dataframe that is clean and allows the algorithms to be implemented. In particular, isolate the variable isFraud from the other variables.
# Note: You'll also need to separate the data into a training set and a test set if you're implementing supervised learning models.

# Pre processing:
# nameOrig, nameDest : Hashing Encoder (lot of categories and binary target)
encoder = ce.HashingEncoder(cols=["nameOrig", "nameDest"], n_components=3)

# Application of Hashing Encoding on the data
df_hash = encoder.fit_transform(df)

df_hash.columns


# type: dummy encoding
def get_ohe_single_var(base, var):
    """
    Apply OneHotEncoder to a single variable in the DataFrame.

    Parameters:
    data: input DataFrame
    var (str): column name

    Returns:
    pd.DataFrame: DataFrame with the original column replaced by One-Hot Encoded columns.
    """
    # Initialize the encoder
    encoder = OneHotEncoder(handle_unknown="ignore")

    # Fit and transform the specified column
    encoded_array = encoder.fit_transform(base[[var]])

    # Create a temporary DataFrame with the encoded columns
    encoded_base = pd.DataFrame(
        encoded_array.toarray(),
        columns=encoder.get_feature_names_out([var]),
        index=base.index,
    )

    # Drop the original column and concatenate the encoded DataFrame
    base = base.drop(columns=[var])
    base = pd.concat([base, encoded_base], axis=1)

    return base


df_hash = get_ohe_single_var(df_hash, "type")

y = df_hash["isFraud"]

X = df_hash.drop(columns=["isFraud"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# (e) Implement several algorithms to solve the fraud detection problem.
# Note: Try out as many algorithms as possible: classification, clustering, local outlier detection algorithm...

# Note 2: Don't hesitate to adjust the algorithms' hyperparameters.

# KNN
neighbours = np.arange(1, 25)
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

for i, k in enumerate(neighbours):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)
    knn.fit(X_train, y_train.ravel())
    train_accuracy[i] = knn.score(X_train, y_train.ravel())
    test_accuracy[i] = knn.score(X_test, y_test.ravel())
idx = np.where(test_accuracy == max(test_accuracy))
x = neighbours[idx]
knn = KNeighborsClassifier(n_neighbors=x[0], algorithm="kd_tree", n_jobs=-1)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

knn_accuracy_score = accuracy_score(y_test, y_pred_knn)
# LOF
lof = LocalOutlierFactor(
    n_neighbors=10, algorithm="auto", metric="euclidean", contamination=0.002
)
y_pred_lof = lof.fit_predict(X.values)
y_pred_lof[y_pred_lof == 1] = 0
y_pred_lof[y_pred_lof == -1] = 1

lof_accuracy = accuracy_score(y.values, y_pred_lof)
# DBSCAN
model = DBSCAN(eps=0.4, min_samples=15)
model = model.fit(X.values)

y_pred_dbscan = model.labels_
y_pred_dbscan[y_pred_dbscan == -1] = 1
y_pred_dbscan[y_pred_dbscan != -1] = 0

dbscan_accuracy = accuracy_score(y.values, y_pred_dbscan)

# (f) Compare their results.
print("#" * 10, "KNN", "#" * 10)
print("Accuracy score:", accuracy_score(y_test, y_pred_knn), end="\n\n")
print("F1 score:", f1_score(y_test, y_pred_knn), end="\n\n")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_knn), end="\n\n")
print("#" * 50)
print("ROC-AUC SCORE:", roc_auc_score(y_test, y_pred_knn), end="\n\n")
print("#" * 50)
print("MCC Score:", matthews_corrcoef(y_test, y_pred_knn), end="\n\n")
print("#" * 50)
print(
    "Precision-recall Score:", average_precision_score(y_test, y_pred_knn), end="\n\n"
)

print("#" * 10, "LOF", "#" * 10)
print("Accuracy score:", accuracy_score(y, y_pred_lof), end="\n\n")
print("F1 score:", f1_score(y, y_pred_lof), end="\n\n")
print("Confusion matrix:\n", confusion_matrix(y, y_pred_lof), end="\n\n")
print("#" * 50)
print("ROC-AUC SCORE:", roc_auc_score(y, y_pred_lof), end="\n\n")
print("#" * 50)
print("MCC Score:", matthews_corrcoef(y, y_pred_lof), end="\n\n")
print("#" * 50)
print("Precision-recall Score:", average_precision_score(y, y_pred_lof), end="\n\n")

print("#" * 10, "DBScan", "#" * 10)
print("Accuracy score:", accuracy_score(y, y_pred_dbscan), end="\n\n")
print("F1 score:", f1_score(y, y_pred_dbscan), end="\n\n")
print("Confusion matrix:\n", confusion_matrix(y, y_pred_dbscan), end="\n\n")
print("#" * 50)
print("ROC-AUC SCORE:", roc_auc_score(y, y_pred_dbscan), end="\n\n")
print("#" * 50)
print("MCC Score:", matthews_corrcoef(y, y_pred_dbscan), end="\n\n")
print("#" * 50)
print("Precision-recall Score:", average_precision_score(y, y_pred_dbscan), end="\n\n")

# (g) Determine the best model to respond to the problem and display the fraudulent online transactions detected by this model.
# KNN performs best, however, it is still not well: Accuracy is high due to a good match of non-Fraud, however,
# the algorithm works not good in detecting fraud. Precision-recall score is just 0.23

df_results = pd.DataFrame({"true_label": y_test, "pred_label": y_pred_knn})

df_results["correct_pred"] = df_results["true_label"] == df_results["pred_label"]

plt.figure(figsize=(12, 9))
sns.scatterplot(
    x=df_results.index,
    y="true_label",
    hue="correct_pred",
    data=df_results,
    palette="coolwarm",
)
plt.title("True against predicted labels")
plt.xlabel("Index Transaction")
plt.ylabel("True Label")
plt.legend(title="Correct Predcition")
plt.show()

# We can clearly see from the graph that the model does not have a good turn on predicting fraud. Only a few points
# on value 1 are in lightred indicating a correct match.
