# Clustering with scikit-learn

# The two main unsupervised classification methods, seen during the training, are hierarchical ascending classification (HAC) and moving center methods (K-Means). However, they both have certain advantages and disadvantages:

# The K-means algorithm: is easy to implement, and applicable to any type and any size of data. However, the number of classes must be fixed a priori, and the final result is dependent on the initial (random) drawing of the centers of classes.

# Hierarchical Ascending Classification (HAC) has the main advantage of being able to visualize the progressive grouping of data as well as the increase in dispersion in a group produced by an aggregation, thanks to the dendrogram. It is not necessary to define the number of classes in advance, and the dendrogram helps to get an idea of ​​the adequate number of classes in which the data can be grouped. However, the CAH requires the calculation of the distances between each pair of individuals, and can therefore be very long as soon as the number of individuals is high (1000+).

# An effective solution to benefit from the advantages of these two methods while reducing their respective disadvantages is to use the method of Mixed Classification.

# It consists of three steps:

# Application of the K-Means method to quickly obtain a fairly large number of homogeneous classes. A good practice is to take a number of groups 10 times lower than the number of observations.

# Ascending hierarchical classification applied to the centroids obtained during the first step using a dendrogram to choose the number of clusters.

# Application of the K-Means algorithm on the entire original dataset using the centroids obtained during the CAH step as initial centroids of the K-Means.

# The objective of the test is to apply a mixed classification algorithm based on the packages and methods studied in the training. No knowledge outside the course is necessary to carry out this step.

# The data used comes from images of various vehicles and contains characteristics specific to their silhouette. The aim of the exercise is to succeed in classifying the vehicles into several groups, according to these characteristics.

# 1. Importing and preparing data

# In this part, we will import the data, modules used in the exercise. In addition, we will explore and clean the data used.

# (a) Run the following command to import the modules needed for this exam.

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt

# (b) Using pandas, create a DataFrame calledsv_datafrom the file"vehicles_silhouette.csv". Determine the column containing the vehicle index.
sv_data = pd.read_csv('vehicles_silhouette.csv', index_col='vehicule_id')

# (c) Display the first 5 rows of the dataset.
sv_data.head()

# (d) Display a description of the various variables using the describe method.
sv_data.describe()

# (e) Using a boxplot, visually compare the distributions of numerical variables.
sv_data.boxplot()
plt.xlabel('Variables')
plt.ylabel('Values')

# (f) Perform a Min-Max normalization on the different variables of sv_data. We can use the MinMaxScaler class from the sklearn.preprocessing sub-module.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

sv_data = pd.DataFrame(scaler.fit_transform(sv_data), columns=sv_data.columns)

# 2. Mixed Classification - 1st KMeans

# In this part, we are going to apply the first step of the mixed classification: we are going to apply a KMeans algorithm on the basic observations.

# (a) Apply a KMeans algorithm on sv_data taking45 clusters.
kmeans = KMeans(n_clusters=45)

kmeans.fit(sv_data)

# (b) Extract the centroids of the clusters produced by this K-Means and store them in a variable named centroid_kmeans1.
centroids_kmeans1 = kmeans.cluster_centers_

# (c) For each sample of sv_data, determine the cluster it belongs to according to the K-Means you just trained. Store product labels in a variable named kmeans1_predictions.
kmeans1_predictions = kmeans.labels_

# (d) Show distribution of observations by cluster. We can use the value_counts method or a histogram. Does the distribution in the different clusters seem balanced to you?
sv_data2 = sv_data
sv_data2['kmeans1_pred'] = kmeans1_predictions

sv_data2['kmeans1_pred'].value_counts()

plt.figure()
plt.hist(sv_data2['kmeans1_pred'], bins=len(sv_data2['kmeans1_pred'].unique()))

# Does not seem balanced. Difference between lowest and highest is nearly 50!

# 3. Mixed Classification - Ascending Hierarchical Classification

# In this part, we are going to apply a hierarchical ascending classification on the centroids from the previous step.

# (a) We now want to group the resulting centroids into clusters, not the samples. To do this, construct a dendogram from centroid_kmeans1 to determine an optimal number of clusters (strictly greater than 2). Assign this number of clusters to a variable named n_clusters_cah.

linked = linkage(centroids_kmeans1, method='ward')

linked 

dendrogram(linked, leaf_rotation = 90., color_threshold = 0)

# for me it is 2 (highest space if I remember correctly ca. 3 to slightly above 5)
# arbitrary set of n_clusters_cah
n_clusters_cah = 2

# (b) Apply a hierarchical ascending clustering on centroid_kmeans1 using the optimal number of clusters found previously.

# (c) For each centroid of centroid_kmeans1, determine the cluster it belongs to according to the CAH you just trained (labels_ attribute). We will store the result in a variable named cah_clusters.

hac = AgglomerativeClustering(n_clusters=2)

hac.fit(sv_data)

cah_clusters = hac.fit(centroids_kmeans1)

cah_clusters.labels_

# We must now associate a cluster from the CAH to each observation of the initial set. For this we must:

# Associate each of the observations of sv_data to a cluster of the initial K-Means.

# Make the correspondence between the K-Means clusters with the CAH clusters to associate a CAH cluster with each observation.

# (d) Using kmneans1_predictions and cah_clusters, create an additional column in sv_data named 'cluster_CAH'. This column will contain the associated CAH cluster.
sv_data['kmeans_cluster'] = kmeans1_predictions

sv_data['cluster_CAH'] = np.nan

for cluster in np.unique(kmeans1_predictions):
    # gather all indices where condition is true, meaning gathering all data points of specific cluster
    km_cluster_ind = np.where(kmeans1_predictions == cluster)[0]
    # find the corresping cah_cluster
    cah_cluster_km = cah_clusters.labels_[cluster]
    # set the value in the new variable
    sv_data.loc[km_cluster_ind, 'cluster_CAH'] = cah_cluster_km
    
sv_data[['kmeans_cluster', 'cluster_CAH']].head()

# We will now be able to calculate the centroids of the ascending hierarchical classification.

# (e) Using the groupby method, compute the centroids of each cluster obtained by CAH, i.e. it will take calculate the average of each column of sv_data based on the value of column "cluster_CAH". We will store the result in a variable named centroids_cah.
centroids_cah = sv_data.groupby('cluster_CAH').mean()

centroids_cah

# (f) Remove column "cluster_CAH" from sv_data.
sv_data = sv_data.drop(columns='cluster_CAH', axis=1)

# 4. Mixed Classification - 2nd KMeans

# In this part, we are going to retrain a KMeans algorithm, specifying this time to take the centroids calculated in the previous step as initial centroids.

# (a) Train a K-Means algorithm on sv_databy passing centroids_cah as an argument to the init parameter of KMeans.Use the optimal number of clusters found during the HAC step.
kmeans = KMeans(init=centroids_cah, n_clusters=2)

kmeans.fit(sv_data)

# (c) For each sample of sv_data, determine the cluster it belongs to according to the K-Means you just trained. Store product labels in a variable named kmeans2_predictions.
kmeans.predict(sv_data)

kmeans2_predictions = kmeans.labels_

sv_data['kmeans2_predictions'] = kmeans.labels_

# 5. Predictions

# In a file named "classes_vehicles.csv", we have the type of vehicle for each observation of sv_data: car, bus or van. We will determine if our clusters are representative of these labels.

# (a) Read the file "classes_vehicles.csv" into a DataFrame named classes and display the first 5 lines .
classes = pd.read_csv('classes_vehicles.csv', sep=';', index_col='vehicule_id')

classes.head()

# (b) Using a crosstab, compare the clusters found through the mixed classification with the vehicle types.Determine which cluster could correspond to each type of vehicle.
combined = sv_data.join(classes)

pd.crosstab(combined['class'], combined['kmeans2_predictions'])

# No it is not the representation of the type of the car.