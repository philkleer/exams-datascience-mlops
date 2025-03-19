# Time Series
# Dataset

# This exercise focuses on the usage of Portland's public transportation network. We have two columns: the date, in yyyy-mm-01 format (measurements are taken on the first day of each month), and the corresponding number of riders (riders). The goal of the exercise will be to analyze these values to find a $\operatorname{SARIMA}$ model that provides satisfactory predictions for a one-year horizon. The steps to follow before applying this $\operatorname{SARIMA}$ model are:

# Data exploration and model choice (additive or multiplicative)
# Data transformation if needed
# Verification of stationarity and determination of the differencing order d and seasonal differencing order D
# Identification of possible orders for p, q, P, and Q.
# (a) Import the libraries pandas, numpy, matplotlib.pyplot, and statsmodels under the names pd, np, plt, and sm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# (b) Load the data contained in the file portland_v2.csv. Pay attention to the index format and the data types before proceeding to the next questions. You can verify that the series is properly indexed by calendar data by displaying the index attribute of the series.
data = pd.read_csv(
    "portland_v2.csv", index_col=[0], header=0, parse_dates=[0], squeeze=True
)

data.head()

# (c) Plot the entire series on a graph.
data.plot()
plt.ylabel("No. of passenger")
plt.xlabel("Year")
plt.show()

# (d) Perform two seasonal decompositions using statsmodels: the first with an additive model and the second with a multiplicative model. Display the corresponding graphs.

# (e) Is there seasonality? If yes, what is the period?

# (f) We aim to choose the model that provides the most stationary residuals. What is this model?

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data)
result_mult = seasonal_decompose(data, model="multiplicative")

print("additive")
result.plot()
plt.show()
print("multiplicative")
result_mult.plot()
plt.show()

# There is a linear time increasing trend although at the end it does not increase much/decrease only a litte.
# However, over all it is increasing. Hence, seasonality is year.
# We see a clear highs and lows in summer and in winter in each year.
# The multiplicative models gives more stationary results, since the residuals are equally distributed.
# For the additive model, the residuals show a lot of white noise.

# (g) Using Numpy, store the logarithm of the series in the variable datalog. Plot the new series on a graph. Why is this manipulation relevant in our case?
datalog = np.log(data)

# it is relevant since we can only analyze additive models and with the log transformation we transform the
# multiplicative model to an additive model with this link function.

# We now aim to verify the stationarity of our series and determine the order of differencing for the simple term, d, as well as for the seasonal term, D. To do this, we will use two approaches: a visual and empirical approach using the function pd.plotting.autocorrelation_plot, and a statistical approach with the Augmented Dickey-Fuller test (ADF).

# (h) Display the autocorrelogram of the datalog series. Should the series be differenced? Why?
pd.plotting.autocorrelation_plot(datalog)

# the decrease is slow for the autocorrelation that's why we need to differentiate.

# (i) Create and display the autocorrelogram of the datalog_1 series, corresponding to the datalog series differenced at order 1. Remember to remove the missing values created by the differencing. Does the series appear stationary?
datalog_1 = datalog.diff().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
datalog_1.plot(ax=ax1)
pd.plotting.autocorrelation_plot(datalog_1, ax=ax2)

# there are still significant peaks, probably due to seasonality. However, it goes further into direction of stationary than before

# We will now deseasonalize the series based on the period identified earlier. For this, we can use the periods parameter of the diff method.

# (j) Create and display the autocorrelogram of the datalog_2 series, corresponding to the datalog_1 series differenced and deseasonalized. Does the series appear stationary?
datalog_2 = datalog_1.diff(periods=12).dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
datalog_2.plot(ax=ax1)
pd.plotting.autocorrelation_plot(datalog_2, ax=ax2)

# now we can see that the peaks within the lines, and therefore, it appears to be stationary.

# (k) Use the Augmented Dickey-Fuller (ADF) test, implemented in the statsmodels library through the adfuller function in the tsa.stattools submodule. Conclude on the stationarity of datalog_2.
sm.tsa.stattools.adfuller(datalog_2)[2]

# we see that the p-value is 0, therefore, we can accept the data as stationary.

# So far, we have differenced the series once with a period of 1 and another time based on seasonality (k), which has stationarized the time series. Thus, we have d = 1 and D = 1. Now, we will visually determine the parameters for the AR and MA models.

# (l) Use the plot_acf and plot_pacf functions from the statsmodels.graphics.tsaplots submodule to plot the simple autocorrelation (ACF) and partial autocorrelation (PACF) of the datalog_2 series. Set lags to 36.
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
plot_acf(datalog_2, lags=36, ax=ax1)
plot_pacf(datalog_2, lags=36, ax=ax2)
plt.show()

# (m) Using the obtained graphs, visually determine the orders p, q, P, and Q.
# Here are the rules to follow for determining the orders:

# For the non-seasonal part (p and q):

# Order p - Look at PACF:

# Count the significant peaks up to the first "hole" (the first peak is only a reference and is not counted).
# Peaks must exceed the dotted lines (confidence intervals).
# p = number of consecutive significant peaks.
# Order q - Look at ACF:

# Same principle: count the significant peaks up to the first "hole".
# q = number of consecutive significant peaks.
# For the seasonal part (P and Q):

# Order P - Look at PACF:

# Observe the peaks at multiples of s=12.
# Count the significant seasonal peaks (12, 24, 36...).
# P = number of significant seasonal peaks.
# Order Q - Look at ACF:

# Observe the peaks at multiples of s=12.
# Count the significant seasonal peaks.
# Q = number of significant seasonal peaks.

# suggestions: p, q = 0 (no significant peak after 1st line at 0)
# suggestions: P = 1 (at 12 significant then only decreasing)
# suggestions: Q = 1 (at 12 significant, after on only insignificant and lowe)

# (n) Instantiate a $\operatorname{SARIMAX}$ model on the datalog series. Arbitrarily use the parameters p=0, d=1, q=0, P=0, D=1, Q=1, and k=12. Display an analysis of the model using the summary method.
model = sm.tsa.SARIMAX(datalog, order=(0, 1, 0), seasonal_order=(0, 1, 1, 12))
sarima = model.fit()
print(sarima.summary())

# (o) Using the get_forecast method, predict the values of the series for the 12 months following the last observed value. Display the predictions and the original series on the same graph. Remember the log transformation applied in the previous questions.
prediction = sarima.get_forecast(steps=12).summary_frame()

fig, ax = plt.subplots(figsize=(15, 5))

prediction = np.exp(prediction)
plt.plot(data)
prediction["mean"].plot(ax=ax, style="k--")
ax.fill_between(
    prediction.index,
    prediction["mean_ci_lower"],
    prediction["mean_ci_upper"],
    color="k",
    alpha=0.1,
)

# The data in the file portland_8182.csv contains the actual values for these 12 months.

# (p) Plot on the same graph the predictions, the actual values, and, if desired, the confidence interval of the predictions.

data_correct = pd.read_csv(
    "portland_8182.csv", index_col=[0], header=0, parse_dates=[0], squeeze=True
)


prediction = sarima.get_forecast(steps=12).summary_frame()

fig, ax = plt.subplots(figsize=(15, 5))

prediction = np.exp(prediction)
plt.plot(data)
prediction["mean"].plot(ax=ax, style="k--", label="Predicted")
ax.fill_between(
    prediction.index,
    prediction["mean_ci_lower"],
    prediction["mean_ci_upper"],
    color="k",
    alpha=0.1,
)

data_correct.plot(ax=ax, style="r-", label="Actual")

plt.title("SARIMA Predictions vs. actual data")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend(loc="upper left")
plt.show()

# The prediction error is defined as: $$X - \widehat{X}$$
# It indicates whether the prediction overestimates or underestimates the actual values of the series.

# The relative mean error is defined as: $$\displaystyle 100 \cdot \overline{\frac{|X - \widehat{X}|}{X}}$$
# It evaluates the quality of the predictions: the lower the percentage, the closer the predictions are to reality.

# (q) Calculate the prediction error and the relative mean error of the prediction.

# (r) Conclude on the model's quality. Does it overestimate or underestimate the data?

rme = 100 * abs((data_correct - prediction["mean"])) / data_correct

print("The relative mean error is", rme.mean())
# the predictions are quite close to the reality (as can also be seen in the graph). The relative mean error is just 1.72%.
