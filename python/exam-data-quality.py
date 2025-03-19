# data quality

# 1. Data Inspection

# (a) Import the file "travel_insurance.csv", and display its first lines.

import pandas as pd
import numpy as np

df = pd.read_csv("travel_insurance.csv")

df.head()

# (b) The column 'Commision' is misspelled. Rename it to 'Commission'.
df.rename({"Commision": "Commission"}, axis=1, inplace=True)

df.head()

# (c) Display information about the variables and their type.
df.info()

# (d) Which 2 variables are assigned the wrong type?
# (e) Modify their type (by making modifications to the data if necessary) by explaining the approach.
# Comission is object, but it is a number from the description
# Date is an object and should be a date.
# Duration is also an object, but should be a number since it is
#  the duration in some kind of unit (minutes, hours or anything else, not described)

# Changing commission
# checking values in commission
print(df.Commission.unique())

# it has nan and ? besides numbers
# changing ? to nan
df.loc[(df.Commission == "?"), "Commission"] = np.nan

print(df.Commission.value_counts(dropna=False))

df["Commission"] = df["Commission"].astype(float)

# other two variables
# Date change with to_date
print(df.Date.unique())
# print(df.Date.value_counts()) #checking any non conform values

df.Date = pd.to_datetime(df["Date"], format="%Y-%m-%d")

df.info()

# Duration: change to integer
df.Duration.unique()
df.Duration.value_counts(dropna=False)

# Some numbers are float with Y at the end, some have wrong spelled NaN
df.loc[23, "Duration"]  # with Y at the end
df.loc[27, "Duration"]  # misspelled NaN

# we change each value with Y to get rid of the Y
for index, value in enumerate(df["Duration"]):
    if not (df.loc[index, "Duration"].isnumeric()):
        # taking the Y way by splitting at that point
        df.loc[index, "Duration"] = value.split("Y")[0]

# Changing wrong Nas to correct NAs
df.loc[(df["Duration"] == "NaNN"), "Duration"] = np.nan

# bringing to float; to bring it to integer, we need to bring NA's first to number
# and then we could bring it back to int
df["Duration"] = df["Duration"].astype(float)

# alternative int
# df['Duration'] = df['Duration'].fillna(-99)
# df['Duration'] = df['Duration'].astype(int)
# df['Duration'] = df['Duration'].replace(to_replace = -99, value = np.nan)

# 2. Data standardization

# The 'Net Sales' variable is a variable representing the amounts of sales of travel insurance policies.

# (a) Display the distribution of the variable and handle outliers if there are any.
# Justify the process.

df[df["Net Sales"] == df["Net Sales"].max()]

df.boxplot("Net Sales")
# There is a huge variety among the variable 'Net Sales' with Agencies selling up to
#  800 insurances. However, these values are valid and should be included in the data,
#  although they are outliers.

# The negative value on Net Sales don't make sense, since an agency canot sell -100 or
#  -150 insurances. Therefore, we put all negative values to np.nan
df.loc[(df["Net Sales"] < 0), "Net Sales"] = np.nan

df.boxplot("Net Sales")

# (b) Display the list of all different destinations by sorting them in alphabetical order.
# (c) Spot an input error and replace the misspelled destination (help: inspect the previous list from the end).
print(df["Destination"].sort_values().unique())

df.loc[(df["Destination"] == "VIET NAM"), "Destination"] == "VIETNAM"

print("\n New List\n")

print(df["Destination"].sort_values().unique())

# (d) Does the dataset contain duplicates? If so, fix this problem.
df.duplicated().sum()

# The data set contains 3534 duplicates

df.drop_duplicates(keep="first", inplace=True)

# (e) Display the distribution of the modalities taken by the variable 'Product Name'.
# (f) Is the variable relevant in your opinion? If not, propose a solution for improvement.
print(df["Product Name"].value_counts())

# There are 26 different plans. Therefore, the variable has relevant information. However, for a specific data analysis it might be better to recode into dummy variables for each  plan.
# However, this apporach would increase the data frame by 24 columns. Therefore, grouping plans into categories might be senseful given a concrete data analysis plan.

# 3. Handling missing values

# (a) Display the number of missing values per column.
df.isna().sum()

# (b) Replace, by the method of your choice, the missing values of the 'Commission' column.
# Since Commission is a float variable, quantitative variable, we set it to 0. Mean would also be an option, but might be highly mis-informative, therefore, I choose to set to 0.
df["Commission"] = df["Commission"].fillna(0)

# (c) Display the frequencies of the values taken by the column 'Gender', including the NAs. Take, with justification, an appropriate decision.
df["Gender"].value_counts(dropna=False)

# There are 46439 missing values on Gender, which reflects 73% of the cases.
# Assuming the  analysis rely on Gender, we could only work with nearly a quarter of the data. Replacing the missing value with one of the values (M, F) would only be possible if we had further  variables that might indicate gender (i.e., names).
# Assuming the analysis does not rely on Gender, the best way would to drop this variable, since the mode is 'NA'.
# We delete the column Gender and save the object into a new object called 'new_df'
new_df = df.dropna(subset=["Gender"], axis=0, how="all")

# (d) Display the distribution of the variable 'Claim'. What do we deduce about the relevance of this variable? Take, with justification, an appropriate decision.
df["Claim"].value_counts(dropna=False)

# There are some NAs and about 95% have a 'No' on this variable. Variation therefore does not
#  really exist and information from this variable seem irrelevant.
# To not loose the information, we save the subsetted df into a new object
new_df = df.drop("Claim", axis=1)

# Another approach might to save the data into a new file
# df['Claim'].to_csv('df-claim.csv', index = True)
# And then delete the column
# df.drop('Claim', axis = 1, inplace = True)
