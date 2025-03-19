# Exploratory Statistics

# 1. Data simulation

# The objective of this first part is to simulate data from a probability law and to study "empirically" their mathematical behaviour.

# (a) Run the next cell to import some packages needed for the exercise. As you go along you may have to load other packages useful for your analysis.

import pandas as pd
import numpy as np
import statsmodels.api as sm

# (b) In a variable simu_norm simulate $1000$ random draws from a normal law of parameters $\mu = 3$ and $\sigma = 0.1$. To do do we can use the Numpy method np.random.normal.
# (c) Display its histogram.

simu_norm = np.random.normal(loc=3, scale=0.1, size=1000)

import seaborn as sns

sns.histplot(simu_norm)

# (d) Give the value of the mean and the standard deviation of the sample.
# (e) Why are these values not equal to the parameters of the law that generated the sample?

print("Mean:", simu_norm.mean())
print("SD:", simu_norm.std())

# They are only approximately equal to the values of generating the sample,
# since the sample itself relies on a random sample strategy.

print(
    "Answer to e): They are only approximately equal to the values of generating the sample, since the sample itself relies on a random sample strategy."
)

# (f) Illustrate (possibly with a loop) that if the sample size increases the values of the (empirical) sample mean and standard deviation approach the theoretical values.

nsizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

for i, size in enumerate(nsizes):
    mu = 3
    sigma = 0.1
    simulation = np.random.normal(loc=mu, scale=sigma, size=size)

    print(
        "Iteration:",
        i,
        ", value of mean:",
        simulation.mean(),
        ", value of SD:",
        simulation.std(),
    )
    print(
        "Iteration:",
        i,
        ", Diff to mu:",
        mu - simulation.mean(),
        ", value of SD:",
        sigma - simulation.std(),
    )

# 2. Analysis on a given dataset

# The dataset "youtube.csv" contains $122391$ youtube videos (per row) which are described by several characteristics: video id, video title, video publication date, channel name, category id, number of views, number of likes and dislikes, number of comments, comments disabled or not for the video.

# (g) Load the dataset "youtube.csv" into a DataFrame named df and display the first five lines.

df = pd.read_csv("youtube.csv")

df.head()

# (h) Display the type of each column and change it if there are variables that are not in a suitable format.

print(df.info())

# changing time stamp variable into time-format
df["publishedAt"] = pd.to_datetime(df["publishedAt"])

print(df.info())

# (i) Display the different modalities of the column "categoryId", as well as their frequency.

df["categoryId"].unique()

# The column "categoryId" is the category of the video. Here is the meaning of each category:

# |Modality | Signification| |------|---------| |1| Film & Animation | |2| Autos & Vehicles | |10| Music | |15| Pets & Animals | |17| Sports | |19| Travel & Events | |20| Gaming | |22| People & Blogs | |23| Comedy | |24| Entertainment | |25| News & Politics | |26| Howto & Style | |27| Education | |28| Science & Technology | |29| Nonprofits & Activism |

# (j) Create a new dataframe named df_filter from df keeping only the categories Entertainment, Music, Gaming and Education. Display the first $5$ lines.
# Note: To filter you can use the |* ("OR") operator or use the .isin([]) command

df_filter = df[(df["categoryId"].isin([24, 10, 20, 27]))]

df_filter.head()

# (k) In this new dataframe, replace the numbers present on "categoryId" with the name of the respective category.

df_filter["categoryId"] = df_filter["categoryId"].replace(
    to_replace=[24, 10, 20, 27], value=["Entertainment", "Music", "Gaming", "Education"]
)

df_filter.head()

# (l) Grouping the data, we can display with a plot the evolution of the average number of likes as a function of the date of appearance of the video (with a monthly frequency) and distinguishing the category of the video. Run the following cell and comment the output.

groupby_m = (
    df_filter.groupby(
        [pd.Grouper(key="publishedAt", freq="m"), df_filter["categoryId"]]
    )
    .agg({"likes": "mean"})
    .unstack()
    .fillna(0)
)

groupby_m.plot(figsize=(20, 4.5), style="o-")
print(
    "Overtime, we can see that videos dealing with music have the highest number of likes. The other three categories have only minor numbers of likes.\n"
)
print(
    'Furthermore, we can see that the number of likes among the categories "Education", "Gaming", and "Entertainment" do not vary a lot over time. Instead, the number of likes for music videos varies greatly over teim from a low 100,000 (Dec 20) to a high 400,000 (Aug 20).'
)
print(
    "For music videos, we can recognize that during the winter months (Dec, Jan, Feb) the average number of likes are lowest in 2021 and 2022."
)
print(
    "For educational videos, the number of likes are the lowest constantly over time among the four categories."
)

# (m) Now we are looking to see if there is a link between the category of the video and the month in which the video appeared. Propose a method to answer this question and conclude.
# Hint: You can create a new column by getting the month from the column "publishedAt" using a specific argument that applies to pandas.Series in datetime.date format

df["publicationMonth"] = df["publishedAt"].dt.month
df.head()

# Chi**2 test: two categorical variables
test_table = pd.crosstab(df["publicationMonth"], df["categoryId"])

from scipy.stats import chi2_contingency

result = chi2_contingency(test_table)

print("Test results:", result[0], "(Test value);", result[1], "(p-value)\n")
print(
    "From the result, we can see that there is a significant relationship between publication months and the category of video."
)

# From the result, we can see that there is a significant relationship between publication months and the category of video.
# (n) From this result, and by running the next cell, can we conclude that there is a hierarchy between the category and the number of times the video appears over time (as a function of month)?

categories = (
    df_filter.groupby(
        [pd.Grouper(key="publishedAt", freq="m"), df_filter["categoryId"]]
    )
    .agg({"video_id": "count"})
    .unstack()
    .fillna(0)
)

categories.plot(figsize=(20, 4.5), style="o-")

print(
    "The only thing that is visible is that videos in the category education constantly get similar counts, independently from the month."
)
print(
    "For the other categories, we can see that the count of videos from the categories entertainment, gaming and music varies over time."
)

# (o) Display the position and dispersion indicators (the most common ones) of the "likes " variable of the df.
df["likes"].describe()

# (p) Find the video that has had the highest number of likes.
print(df[(df["likes"] == df["likes"].max())])

# (q) From df create a df_num dataframe that contains only columns with numeric data and display the first $5$ rows.

df_num = df.select_dtypes(include=["int", "float"])

df_num.head()

# (r) Calculate the correlation coefficients between the different columns.

df_num.corr()

# (s) Does the data in the view_count variable look normal?
# Hint : To answer this question, use a graphical function in the statsmodels.api library.

import statsmodels.api as sm

sm.qqplot(df["view_count"], fit=True, line="45")

print(
    'The distribution of "view_count" absolutely does not follow a normal distribution. '
)

# (t) We would like to know if disabling comments on YouTube has an influence on the number of likes.
# Propose a scientific approach, describe the steps and apply the specific method used, then conclude.

# Disabling comments: categorical; number of likes: quantitative => ANOVA
# We choose ANOVA since one variable is quantitative and the other is categorical

# Importing the relevant library
import statsmodels.api

# calculating the test
result = statsmodels.formula.api.ols("likes ~ comments_disabled", data=df).fit()

# Printing the results
print(statsmodels.api.stats.anova_lm(result))

print(
    "Conclusion, the p-value is lower than 0.05, therefore, we reject the Nullhypothesis. There is a significant difference between the videos enabled comments and disabled comments regarding the amount of likes "
)

print("\n\n")
print(
    "Mean for disabled comments:", df["likes"][(df["comments_disabled"] == True)].mean()
)
print(
    "Mean for enabled comments:", df["likes"][(df["comments_disabled"] == False)].mean()
)
