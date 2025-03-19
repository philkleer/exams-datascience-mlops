# Dimension Reduction
# The exercise is composed of several questions, do them in order and be careful to respect the names of the variables. Do not hesitate to contact the DataScientest team if you encounter any problems.

# The purpose of this exercise is to significantly reduce the dimension of a dataset used for a classification problem. The packages used will be numpy, pandas, matplotlib, scikit-learn and its subpackages. A brief description of the dataset is given below.

# Dataset

# The dataset is based on an experiment with 30 volunteers aged between 19 and 48 years. For each observation, one person performed one of six activities (walking, stair climbing, stair descending, sitting, standing, lying down) while wearing a smartphone around their waist.

# The dataset was constructed using smartphone sensors and mathematical transformation (jerk, ...).

# The objective of the exercise is to predict activity as a function of 562 explanatory variables. Because of the large number of explanatory variables, a selection and a dimension reduction is necessary.

# Run the following cell to import the packages needed for the exercise.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("activity.csv")
df.head()

# (c) Store in a variable named target, the target variable 'activity'.
# (d) Store in a variable called df_new, the df variable without the 'activity' column.
target = df["activity"]
df_new = df.drop(columns="activity", axis=1)

df_new.describe()

# (e) Split the data into two samples: a training sample and a test sample. We will take an 80% distribution on the training data and instantiate a randomised seed at 1234. We will name the data sets X_train, Y_train, X_test and Y_test.
X_train, X_test, y_train, y_test = train_test_split(
    df_new, target, test_size=0.2, random_state=1234
)

X_train.shape

# (f) Using the VarianceThreshold instance, eliminate all explanatory variables that have a variance less than 0.01.
# (g) Transform the training set and store it in a X_train_var variable. Do the same for the test set and store it in a X_test_var.
# (h) Display the selector mask.
sel = VarianceThreshold(threshold=0.01)

X_train_var = sel.fit_transform(X_train)
X_test_var = sel.transform(X_test)

mask = sel.get_support()
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("Features axis")

# (i) Normalise using the StandardScaler instance, the X_train and X_test variables, do the same for the X_train_var and X_test_var variables with a new instance of StandardScaler.
scaler1 = StandardScaler()
scaler2 = StandardScaler()

X_train_scaled = scaler1.fit_transform(X_train)
X_test_scaled = scaler1.transform(X_test)

X_train_var_scaled = scaler2.fit_transform(X_train_var)
X_test_var_scaled = scaler2.transform(X_test_var)

# (j) Fit a k nearest neighbour classifier with k=6 on the training set X_train and Y_train.
# (k) Calculate the model score on the train and test sets.
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train, y_train)

knn.score(X_test, y_test)

# (l) Fit a k nearest neighbour classifier with k=6 on the training set X_train_var and Y_train.
# (m) Calculate the score of the model on X_test_var. What do you notice?
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train_var, y_train)

knn.score(X_test_var, y_test)

# The score value is exatly the same.

# (n) Perform a PCA on the training set X_train_var, without specifying the number of components. Store the results of this transformation in a new variable X_train_pca.
# (o) Display the cumulative explained variance. Determine graphically the number of components retained to keep 95% of the explained variance.

pca = PCA()

X_train_pca = pca.fit_transform(X_train_var)

plt.figure()
# added because it's close to 100 where the threshold is met
plt.xlim([1, 100])
plt.xticks(range(1, 100, 5))
plt.xlabel("Number of components")
plt.ylabel("Explained variance ratio")
plt.axhline(y=0.95, color="r", linestyle="--")
# added line where I assume
plt.axvline(x=66, color="b", linestyle="-")
plt.plot(pca.explained_variance_ratio_.cumsum())

# Depending on the number of principal components chosen earlier :

# (p) Perform a new PCA on the X_train_var training set and store the results in the X_train_pca variable. Transform the data from X_test_var using the previously instantiated PCA and store the results in the variable X_test_pca.
# (q) Plot the scatter plot of the data in the reduced dimensional space of the training set, colouring the points according to their class. Do the same for the test set.
pca = PCA(n_components=66)

X_train_pca = pca.fit_transform(X_train_var)

X_test_pca = pca.transform(X_test_var)

# Scatterplot: 2d => 2 first components
pc1_train = X_train_pca[:, 0]
pc2_train = X_train_pca[:, 1]

pc1_test = X_test_pca[:, 0]
pc2_test = X_test_pca[:, 1]

plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.scatter(pc1_train, pc2_train, c=y_train, alpha=0.7)
plt.title("Training Set")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Class")

plt.subplot(1, 2, 2)
plt.scatter(pc1_test, pc2_test, c=y_test, alpha=0.7)
plt.title("Training Set")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Class")

plt.show()

# (r) Fit a k nearest neighbour classifier with parameter k=6 on the reduced dimensional training set.
# (s) Compute the prediction score on the new test set.
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train_pca, y_train)

knn.score(X_test_pca, y_test)

# (t) Perform a PCA on X_train_var that keeps 200 components.
# (u) Calculate and store in a list the prediction score of the model keeping only the first k components of the PCA. The score will be calculated for 𝑘∈[1:200] .
# (v) Plot the prediction score of the model as a function of the evolution of the dimension reduction.
pca = PCA(n_components=200)

X_train_pca = pca.fit_transform(X_train_var)

X_test_pca = pca.transform(X_test_var)

scores = []

for k in range(1, 201):
    X_train_k = X_train_pca[:, :k]
    X_test_k = X_test_pca[:, :k]

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train_k, y_train)

    score = knn.score(X_test_k, y_test)

    scores.append(score)

print("Predictions scores calculated for k=[1, 200]:", scores)

# (w) What is the maximum score and for what reduction?

best_score = max(scores)
best_k = scores.index(best_score) + 1

print(
    "The best k is",
    best_k,
    "with the best score",
    round(best_score, 5),
    ". Reduction is",
    X_test_var.shape[1] - best_k,
)
