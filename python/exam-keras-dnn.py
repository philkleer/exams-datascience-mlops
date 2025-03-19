# Regression with Keras

# The purpose of this exercise is to compare different regression models capable of predicting the fuel consumption of a car based on some of these characteristics. The target variable is MPG which is the number of miles you can get from a barrel of gasoline. This notebook uses the classic Auto MPG dataset which contains examples of cars from the late 1970s and early 1980s with data such as: engine capacity, power, weight...

# Load the packages and import the data

# Run the following cells to load the necessary packages and import the dataset.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

dataset = pd.read_csv(
    "auto-mpg.data",
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)

# Display the first few lines of the dataset.
dataset.head()

# Clean up the data

# Does the dataset contain missing data? If so, process them according to their amount and position.
dataset.info()

# There are 6 missing values in horsepower, we will impute them with the mode of horsepower

dataset["Horsepower"] = np.where(
    dataset["Horsepower"].isna(), dataset["Horsepower"].mode()[0], dataset["Horsepower"]
)
# The "Origin" column is categorical, with index 1 corresponding to the USA, index 2 to Europe and index 3 to Japan. Transform this variable into 3 binary variables representing the origin of the vehicle for these regions of the world.
dummies = pd.get_dummies(dataset["Origin"], prefix="Origin")

dataset = pd.concat([dataset.drop(columns="Origin", axis=1), dummies], axis=1)

dataset.head()

# Data inspection and preprocessing

# Using, for example, the pairplot or heatmap functions of seaborn, investigate the relationships between the variables in your dataset. What can you deduce from this in the context of our regression problem?
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", center=0)

# we can see probably problems since there are 3 highly correlated variables:
# - cylinders, displacement, horsepower and weight

# Study the distribution of variables in the dataset, why do you think a normalisation is necessary here?
# Except for modelyear, acceleartion, none of the variables shows a normal distribution. However there are no
# high outliers, but we see various local maxima for cylinders, displacement and horsepower
# Min-Max-Scaling could solve this.

sns.pairplot(data=dataset, diag_kind="kde")

# Separate the explanatory variables into a X dataframe and the target variable into y. Then apply a minmax normalisation to your dataset.
X = dataset.drop("MPG", axis=1)
y = dataset["MPG"]

# We will do the scaling after train-test split to avoid data leakage

# Prepare the data for training

# Separate the dataset into training and test sets, the test set will contain 20% of the data.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7786
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# First model

# We have seen in the course that a neuron without an activation function behaves like a linear regression model, so we will implement a linear regression model using Keras.

# Instantiate an Input layer, with the number of explanatory variables in the model as a dimension.

# Instantiate a Dense layer, with 1 neuron.

# Apply the Dense layer to the Input layer.

# In a variable called linear_model define a model using the Model function with the inputs and the outputs arguments being the previous results

# Define in a linear_model variable the model using the Model function and using for the inputs and outputs arguments the results obtained previously.

# Finally, display the model summary

inputs = Input(shape=(9,), name="Input")

dense1 = Dense(units=1, activation="tanh", name="Layer_1")

outputs = dense1(inputs)

linear_model = Model(inputs=inputs, outputs=outputs)

linear_model.summary()

# Compile the model with a loss function: "mean_absolute_error", appropriate to a regression problem and with an Adam optimizer using a learning_rate of 0.1.
# You can define a custom learning_rate by creating a variable: opt = tf.optimizers.Adam(learning_rate=...) which you call when compiling your model.

linear_model.compile(
    loss="mean_absolute_error",
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    metrics=["accuracy"],
)

# Train the model on X_train and y_train using the fit method with the parameters: epochs=100, batch_size=32 and validation_split=0.2.

# Store the training history in a linear_history variable.

linear_history = linear_model.fit(
    X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2
)

print(linear_history.history.keys())

# The keys are present, howver, they are not plotted


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)


plot_loss(linear_history)

# The results are stored on the test set for later.

test_results = {}

test_results["linear_model"] = linear_model.evaluate(X_test, y_test, verbose=0)
# Correcting to just take the MAE value for the later task
test_results2 = {}

test_results2["linear_model"] = linear_model.evaluate(X_test, y_test, verbose=0)[1]

# Second model

# We can now implement a real Deep Learning model and compare the results.

# Instantiate an Input layer, with the number of explanatory variables in the model as a dimension.

# Instantiate two Dense layers, with 64 neurons each and a relu activation function.

# Instantiate the Dense prediction layer, with 1 neuron.

# Apply the layers successively to the inputs, we always use the functional construction seen in the module.

# Define in a variable dnn_model the model using the Model function, using for the inputs and outputs arguments the results obtained previously.

# Display the model summary

inputs = Input(shape=(9,), name="Input")

dense1 = Dense(units=64, activation="relu", name="Layer_1")
dense2 = Dense(units=64, activation="relu", name="Layer_2")
dense3 = Dense(units=1, activation="relu", name="Layer_3")

x = dense1(inputs)
x = dense2(x)
outputs = dense3(x)

dnn_model = Model(inputs=inputs, outputs=outputs)

dnn_model.summary()

# Compile and train the model in the same way as the previous model except that you will use a learning rate of 0.001 and store the training history in a dnn_history variable.
dnn_model.compile(
    loss="mean_absolute_error",
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

dnn_history = dnn_model.fit(
    X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2
)

# Let's retrieve the results with the following line of code.

test_results["dnn_model"] = dnn_model.evaluate(X_test, y_test, verbose=0)[1]

# Performance

# Let's compare the performance of the linear model and the neural network.

pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T

# How do you interpret these results?

# Both values are 0.0 which. means they are perfectly accurate (I think), kind of a perfect model with no errors.
# Probably this indicates a problem of overfitting

# Prediction

# To conclude, you can observe with the following 2 cells the differences between the predictions of the deep learning model and the true MPG values.

test_predictions = dnn_model.predict(X_test).flatten()

a = plt.axes(aspect="equal")
plt.scatter(y_test, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# The model normally predicts reasonably well, we can also observe the error distribution.

error = test_predictions - y_test
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
