# Pipeline

# The objective of this exam is to build a pipeline to clean up and prepare data in order to estimate the response and mobilization times of the London Fire Brigade. This evaluation is based on the "firefighter_london.csv" dataset. This dataset contains the details of each incident, with information on the date and location of the incident as well as its type. The variable AttendanceTimeSeconds corresponds to the difference between the departure and arrival time of the firefighters.

# (a) Load the packages pandas and numpy under their usual aliases.
# (b) Read the file "firefighter_london.csv" into a DataFrame called df.
# (c) Display an overview and a first description of the variables of the dataset.

import pandas as pd
import numpy as np

df = pd.read_csv("firefighter_london.csv", index_col=0)

print(df.head(), end="\n\n")
print(df.info(), end="\n\n")
print(df.describe(), end="\n\n")

# (d) Separate the explanatory variables in a dataframe X and the target variable AttendanceTimeSeconds in y.
X = df.drop(columns="AttendanceTimeSeconds", axis=1)
y = df["AttendanceTimeSeconds"]

# 1. Formatting dates

# In the DataFrame df, there are 2 variables which correspond to dates: 'DateAndTimeMobilised', 'DateAndTimeOfCall'. We will try to extract useful information from each of them: the month, the day of the week, the day and the hour.

#   The function pd.to_datetime() transforms the type of a series passed in argument, into a time series (of type datetime). It is then possible to retrieve partial information from a date of datetime type, such as the year or the month, using the attributes dt.year or dt.month.
# (a) Import the classes BaseEstimator and TransformerMixin from the submodule sklearn.base.

# (b) Store the date columns in a variable date_x.

# (c) Define a class DateFormatter which inherits from the classes BaseEstimator and TransformerMixin. This class will have three methods, __init__, fit and transform, defined as follows:

# __init__: does nothing,
# fit : takes 2 arguments, a DataFrame X and a list y, but does nothing,
# transform :
# takes as argument a DataFrame X,
# initiates a new empty DataFrame new_X,
# iterates all the columns of the DataFrame X and for each column, converts the column into datetime format, extracts the month, the day of the week, the day and the hour and adds them in the new DataFrame new_X,
# returns new_X.

from sklearn.base import BaseEstimator, TransformerMixin

date_x = df[["DateAndTimeMobilised", "DateAndTimeOfCall"]]


class DateFormatter(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = pd.DataFrame()
        for col in X.columns:
            X[col] = pd.to_datetime(X[col])
            new_X[f"{col}_month"] = X[col].dt.month
            new_X[f"{col}_weekday"] = X[col].dt.dayofweek
            new_X[f"{col}_day"] = X[col].dt.day
            new_X[f"{col}_hour"] = X[col].dt.hour

        return new_X


# (d) Run the next cell to verify that the date formatting worked.
date_encode = DateFormatter()

date_encode.fit_transform(date_x)

# 2. Pipeline for data encoding

# Now, we are going to create a pipeline allowing to encode categorical variables.

# (a) Import:

# the class Pipeline and the class FeatureUnion from the submodule sklearn.pipeline,
# the transformers OneHotEncoder and PolynomialFeatures from the submodule sklearn.preprocessing,
# the transformer TargetEncoder from the module category_encoders.
# (b) Instantiate:

# a transformer TargetEncoder and name it te,
# a transformer OneHotEncoder and name it ohe,
# a transformer PolynomialFeatures which will have to calculate polynomial features of order 2 and name it poly.
# (c) Store the following categorical variables in a variable cat_x:  PropertyCategory, PlusCode_Description, IncGeo_BoroughName and CodeStation.

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from category_encoders import TargetEncoder

te = TargetEncoder()
ohe = OneHotEncoder()
poly = PolynomialFeatures(degree=2)

cat_x = df[
    ["PropertyCategory", "PlusCode_Description", "IncGeo_BoroughName", "CodeStation"]
]

# (d) Create a pipeline named poly_encoding with 2 steps:

# Data encoding with OneHotEncoder.
# Calculation of polynomial features.
# (e) From the pipeline poly_encoding and from TargetEncoder, define an object FeatureUnion named cat_pipeline to perform the two encoders in parallel.

poly_encoding = Pipeline(steps=[("onehot", ohe), ("polynomial", poly)])

cat_pipeline = FeatureUnion([("poly_enc", poly_encoding), ("target_enc", te)])

# (f) Run the following cell to verify that your pipeline has been implemented.
f = cat_pipeline.fit(cat_x, y)

# 3. Distance from the fire station

# There are four variables :Longitude_x,Latitude_x,Longitude_y, Latitude_y which correspond respectively to the coordinates of the fire stations and the accident sites. We will use these variables to calculate the distance between the two places thanks to the following formula:

# $$ \sqrt{{(X_{\text{Longitude_x}} - X_{\text{Longitude_y}})}^2 + {(X_{\text{Latitude_x}} - X_{\text{Latitude_y}}})^2 }$$
# (a) Store in a DataFrame dist_x the variables Latitude_x, Longitude_x, Latitude_y and Longitude_y.
dist_x = df[["Latitude_x", "Longitude_x", "Latitude_y", "Longitude_y"]]

# (b) Define a class DistanceFromStation which inherits from the classes BaseEstimator and TransformerMixin. This class will have three methods, __init__, fit et transform, defined as follows:
# __init__: does nothing,
# fit: takes 2 arguments, a DataFrame X and a list y, but does nothing,
# transform :
# takes as argument a DataFrame X,
# calculates the formula defined above,
# returns it in a  DataFrame.


class DistanceFromStation(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["DistanceFromStation"] = np.sqrt(
            (X["Longitude_x"] - X["Longitude_y"]) ** 2
            + (X["Latitude_x"] - X["Latitude_y"]) ** 2
        )

        return X


# (c) Run the following cell to verify that your class has been implemented.
distance_encode = DistanceFromStation()

distance_encode.fit_transform(dist_x)

# 4. Column Transformer

# The pre-processing phase is soon over, two steps are missing:

# the deletion of unnecessary columns,
# the replacement of missing values.
# A single column contains missing values:  'NumPumpsAttending'. We will replace them with the most present value in the column.

# (a) Import the class SimpleImputer from the submodule sklearn.impute.

# (b) Instantiate a transformer SimpleImputer and name it imputer with the argument strategy = "most_frequent"

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")

# We will now define a ColumnTransformer to put together all the pre-processing steps.

# (c) Import the class ColumnTransformer from the submodule sklearn.compose.

# (d) Define a ColumnTransformer called encoding with 5 steps:

# date formatting with date_encode,
# data encoding with cat_pipeline,
# encoding of the station-accident distance with distance_encode,
# deletion of unnecessary columns: 'IncidentNumber', 'NomStation',
# replacement of missing values in the column 'NumPumpsAttending'.

from sklearn.compose import ColumnTransformer

encoding = ColumnTransformer(
    transformers=[
        ("date", date_encode, ["DateAndTimeMobilised", "DateAndTimeOfCall"]),
        (
            "categorical",
            cat_pipeline,
            [
                "PropertyCategory",
                "PlusCode_Description",
                "IncGeo_BoroughName",
                "CodeStation",
            ],
        ),
        (
            "distance",
            distance_encode,
            ["Latitude_x", "Longitude_x", "Latitude_y", "Longitude_y"],
        ),
        ("dropping", "drop", ["IncidentNumber", "NomStation"]),
        ("imputing", imputer, ["NumPumpsAttending"]),
    ]
)

# (e) Run the following cell to verify that your ColumnTransformer has been implemented.
encoding.fit_transform(X, y)

# 5. Final pipeline

# Last step, the dataset is now cleaned, we will be able to make a selection of variables and train a first regression model.

# (a)

# Import SelectKBest from the submodule sklearn.feature_selection.

# Import LinearRegression from the submodule sklearn.linear_model.

# Import StandardScaler from the submodule sklearn.preprocessing.

# Instantiate a transformer SelectKBest and name it selector with the argument k=100.

# Instantiate a transformer StandardScaler() and name it scaler with the argument with_mean=False.

# Instantiate a LinearRegression model and name it model.

# (b) Create a pipeline named final_pipeline with 3 steps:

# data encoding with the ColumnTransformer encoding defined above,
# data normalization with scaler,
# selection of variables with selector,
# application of the linear regressionmodel model.

from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

selector = SelectKBest(k=100)
scaler = StandardScaler(with_mean=False)
model = LinearRegression()
final_pipeline = Pipeline(
    steps=[
        ("transform variables", encoding),
        ("scaling", scaler),
        ("selecting", selector),
        ("modeling", model),
    ]
)

# (c) Separate data into a training set and a test set (20%).
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7786
)

# (d) Run the following cell to verify that the final pipeline has been implemented. (Note: the score is not good but the main goal here is the successful implementation of a Pipeline).
final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)
final_pipeline.score(X_test, y_test)
