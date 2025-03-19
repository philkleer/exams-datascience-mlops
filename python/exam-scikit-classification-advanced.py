# Advanced classification and methodology

# The concept of loans between individuals is presented as an alternative to traditional bank financing. Appeared a few years ago in the United States, it belongs to what is called participatory finance, the famous crowdfunding. Thus, via a platform specially dedicated to this type of financing, it is now possible to free oneself from traditional banking intermediaries.

# An online lending platform between individuals provides potential investors with some of the data it holds on borrowers. The latter are attributed an interest rate that varies according to a credit score. The higher the coast (more reliable and less risky loan) the lower the interest rate.

# From an investor's point of view, loans with higher interest rates are more attractive because they offer a higher return on investment (ROI), but on the other hand, they present risks of not being refunded at all (default).

# Therefore, a prediction model based on Machine Learning, which could predict which loans are most likely to be repaid, especially for high interest rates, would bring a higher return important for investors, by minimizing the associated risks.

# The file "lending_file_2015_2016.csv" contains information on loans requested and accepted in 2015 and 2016. Some variables date from the time of the loan application, others were added after. A description of the different variables is available here.

# (a) Import the database contained in the file"lending_file_2015_2016.csv", and make a first exploration.
import pandas as pd
import numpy as np

data = pd.read_csv("lending_file_2015_2016.csv", index_col=0)

data.head()

# (b) Delete the columns that have more than 40% missing values.
# print missing per column
data.isna().mean() * 100

# delete column if over 40%
for column in data.columns:
    if data[column].isna().mean() > 0.40:
        data.drop(columns=column, axis=1, inplace=True)
        print("Column:", column, " was removed.")

# The column emp_title informs us about the profession of the person who requested the loan.

# (c) Display the number of unique values ​​in the emp_title column. Can we transform this column? Should we remove it?
# (d) Transform or delete the column.

data["emp_title"].nunique()

data["emp_title"].value_counts(normalize=True)

# This high number might suggest that there is no data cleaning at all (like grouping in job areas).
# Since this would be too cumbersome in a 2hrs exam, we will remove the variable

# The variable has 201.347 unique values. The highest value for a job title is 2.04% (teacher)
# of the data. Therefore, at this point, I decide to delete the variable.
data.drop(columns="emp_title", axis=1, inplace=True)

# (e) Remove the rows that have at least one null value.
# Removing all rows with min. 1 Null/NA
data.dropna(axis=0, inplace=True)

# The goal of this exercise is to put yourself in the shoes of a potential investor who wants to know whether or not a loan will be repaid by the borrower in advance.

# The variable loan_status contains the current status of the loans. Each possible status is described below:

# Current : The loan is still current and all payments are up to date.

# In Grace Period: The loan is still current and all payments are up to date.

# Late (16-30 days): The loan is 16 to 30 days past due.

# Late (31-120 days): The loan has been past due for 31 to 120 days.

# Fully Paid: The loan has been fully repaid, either at the end of the three or five-year term or through early repayment.

# Default: There will be no payment for an extended period of time.

# Charged Off: A loan for which future payments are no longer expected to be reasonable.

# (f) Display the loan proportions for each status type.

# (g) You will justify what, in your opinion, are the statuses corresponding to the loans:

# that will be reimbursed: situation A
# that will not be reimbursed: situation B
# for which we cannot make a reasonable assumption: situation C

# I think the description of `In Grace Period` is wrong, since it is the same as for `Current`.
# Usually `In Grace Period` means that a loan is in a pause of payments (although all payments until then were
# on time), which might indicate a risk.
data["loan_status"].value_counts(normalize=1)

# We can see that nearly 63% of the loans are fully paid,ca. 20% are current, ca. 17% are Charged Off and below
# 1% for each of the other categories.

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=data["loan_status"])
plt.show()

# Situation A: reimbursed loan
# this is true for Fully Paid and Current

# Situation B: will not be reimbursed.
# This might get true for Late (31-120 days), Late (16-30 days), Charged Off and In Grace Period

# Situation C: cannot make reasonable assumptions.
# This is true for default, which gives us no information.

# (h) We want to keep only the rows for which we believe the credit will be repaid (situation A) or will not be repaid (situation B) based on the variable current_loan_standing. Remove the rows for which we can't make a reasonable assumption: Scenario C.

# (i) Create a variable called target with the following values:

# 0: situation A
# 1: situation B
# (l) Show the target proportions. What is the type of this Machine Learning problem in terms of the modalities of the variable target?

# filter for the cases mentioned in h)
data = data.loc[
    (data["loan_status"] == "Fully Paid")  # A
    | (data["loan_status"] == "Current")  # A
    | (data["loan_status"] == "Charged Off")  # B
    | (data["loan_status"] == "In Grace Period")  # B
    | (data["loan_status"] == "Late (31-120 days)")  # B
    | (data["loan_status"] == "Late (16-30 days)")  # B
]

# create the target variable
data.loc[:, "target"] = data["loan_status"].apply(
    lambda x: 1
    if x
    in ["Charged Off", "In Grace Period", "Late (31-120 days)", "Late (16-30 days)"]
    else 0
)

data.target.value_counts(normalize=True)

sns.countplot(x=data["target"])
plt.show()

# We can now delete the variable loan_status.

# (m) delete the variable loan_status
data.drop(columns="loan_status", axis=1, inplace=True)

# The variable grade contains a grade from A to G, the interest rate is directly calculated from this grade. A poorly rated but properly repaid loan therefore earns investors more than an A-rated loan.

# (n) Display the proportion of loans in class 0 and class 1 for each of the grades in the grade variable.
pd.crosstab(data["target"], data["grade"], normalize=0)

# (o) Change the values of the variable grade from A to G by grades from 6 to 0 respectively.
data.loc[:, "grade"] = data["grade"].apply(
    lambda x: 6
    if x == "A"
    else 5
    if x == "B"
    else 4
    if x == "C"
    else 3
    if x == "D"
    else 2
    if x == "E"
    else 1
    if x == "F"
    else 0
)

# (p) Choose the observations that you believe are important in order to maximize investor returns and explain your strategy (a poorly rated but properly repaid loan earns more for investors than an A rated loan).
# First, we can take a quick view on differences in interest rate, loan amount,
#  and grade of the loan by target
print(data.groupby("target")["int_rate"].mean())
print(data.groupby("target")["loan_amnt"].mean())
print(
    data.groupby("target")["grade"]
    .value_counts(normalize=True)
    .sort_index(level="grade")
)
# We can see that interest rates are on average 3%points higher and the amount is
#  also slightly higher (1000 US-$).
# However, we can also see that the percentage of being in lower grades is higher for loans that have risks


# Second we can take a look at grade: as suggested above, the interest rate is directly derived from grade
print(data.groupby("grade")["int_rate"].mean())

# The differences are large, to increase returns it might be better to abstain
#  from highest grades
print(data.groupby("grade")["loan_amnt"].mean())
# Grade is also related clearly to loan amount, indicating we might get more
#  return from lower scored grades since the amount is higher and as shown above
#  the interest rate is higher

# Second we can look at observations that are still good with their payments
#  (main criteria) and how interest rates, amounts look in these cases.
# Filter loans that are fully paid
paid_loans = data.loc[data["target"] == 0]

# Compare average interest rates by grade and convert to DataFrame
avg_interest_by_grade = paid_loans.groupby("grade")["int_rate"].mean()

# Find lower-grade loans with high interest rates
important_loans = paid_loans[
    (paid_loans["grade"].isin([0, 1, 2]))
    & (paid_loans["int_rate"] > avg_interest_by_grade.mean())
]

# Review important loans
print(
    important_loans[["grade", "int_rate", "loan_amnt", "term"]]
    .sort_values(by=["int_rate", "loan_amnt"], ascending=False)
    .head(20)
)

# We can see that the ones with highest interest rate and loan_amt graded by 0,
# but are still in payment plans, which means with these we would get a better
# return.

# (q) Remove explanatory variables that do not appear to be relevant.
data.info()

# Checking on categorical data
from scipy.stats import chi2_contingency

print(data.term.value_counts())
print(data.application_type.value_counts())
# dichotom: term, application type

print(data.emp_length.value_counts())
print(data.home_ownership.value_counts())
print(data.verification_status.value_counts())
print(data.purpose.value_counts())
# polytom: emp_length, home_ownership, verification_status, purpose

# Categorical, dichotom variable
# checking statistical relationship to target

# term
stat, p = chi2_contingency(pd.crosstab(data["target"], data["term"]))[:2]

V_Cramer = np.sqrt(stat / pd.crosstab(data["target"], data["term"]).values.sum())

print("V Cramer is: ", V_Cramer)
print("p value is: ", round(p, 5))
# variable is relevant and need to be kept1

# application_type
stat, p = chi2_contingency(pd.crosstab(data["target"], data["application_type"]))[:2]

V_Cramer = np.sqrt(
    stat / pd.crosstab(data["target"], data["application_type"]).values.sum()
)

print("V Cramer is: ", V_Cramer)
print("p value is: ", round(p, 5))
# variable is not relevant and can be deleted

# categorical polytom variables:
for column in ["emp_length", "home_ownership", "verification_status", "purpose"]:
    for i in pd.get_dummies(data[column]):
        # Chi-Square test
        stat, p = chi2_contingency(
            pd.crosstab(data["target"], pd.get_dummies(data[column])[i])
        )[:2]

        # Cramer's V
        V_Cramer = np.sqrt(
            stat / pd.crosstab(data["target"], data[column]).values.sum()
        )

        # Only significant variables with a Cramer's V greater than 0.1 are displayed.
        if p < 0.05:
            if V_Cramer > 0.1:
                print(column, i, V_Cramer, "PASSED significance & relevance.")
            else:
                print(column, i, "has not reached relevance")
        else:
            print(column, i, "has not reached significance.")

# none of the four polytom categorical ariables has reached statistical significance (p< 0.05) and
# relevance (Cramers V>0.1), therefore, I delete all of them.
# furthermore, I decided to delete application_type since both, cramer's V and significance are again above thresholds

data.drop(
    columns=[
        "application_type",
        "emp_length",
        "home_ownership",
        "verification_status",
        "purpose",
    ],
    axis=1,
    inplace=True,
)

# Continuous variables

# First we check correlation between explanatory variables. We already know that grade and int_rate correlate high,
# since the later is built up on the first.
# We check if there is more high correlations and then eliminate.
num_vars = data.select_dtypes(include="number")
target = num_vars["target"]

# delete target and id variable from numerical df
num_vars = num_vars.drop(columns=["id", "target"], axis=1)


sns.heatmap(data=num_vars.corr(), annot=True, cmap="rainbow")
plt.show()
# We see the following risky correlations:
# - grade and int_rate (as already assumed)
# - loan_amnt correlates high with installment and total_pymnt; also total_pymnt correaltes high with installment

# For these variables, we need to delete some to not get problems in the calculations.

# Furthermore, we need to delete: next_pymnt_d and total_pymnt since they have obtained some time after the start
# of the loan


# to delete: next_pymnt_d, total_pymnt

# installment and loan_amnt express the same thing: the first is monthly payments, the last is the overall amount
# we keep the one with more relation to the target
# Anova test
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm

attributes = data[["installment", "loan_amnt"]]
results = {}
feat_select = []

for i in attributes:
    lm = ols("target ~ {}".format(i), data=data).fit()
    table = anova_lm(lm)
    liste_p_value = table["PR(>F)"].iloc[:1].to_list()
    results[i] = liste_p_value

    if liste_p_value[0] <= 0.5:
        feat_select.append(i)
results

# Both are similar significant, therefore we keep loan_amnt as overall value.

# From grade and interest, we keep interest, since it represents a numerical variable and not a categorical
# ordered one as grade

# next_pymnt_d does not exist
data.drop(columns=["total_pymnt", "grade", "installment"], axis=1, inplace=True)
data.info()

# delete id variable
data.drop(columns=["id"], axis=1, inplace=True)

# save cleaned file
data.to_csv("data-cleaned.csv", index=False)

# We begin the modeling process by:

# (r) Carry out the required pre-processing steps.

# (s) Create a training and a test set.

# (t) Choose and justify a metric related to your strategy.

# (u) Train and fine-tune the chosen model (s). Examine and comment on your findings.

from sklearn.model_selection import train_test_split

# First i split training and test set, so that there is no information leak in pre-processing steps.
X = data.drop(columns="target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# checking distribution of quantitative variables, for choosing right scaling.
def var_quant_anal(variable, base):
    print(base[variable].describe(), end="\n\n")
    plt.figure()
    sns.displot(data=base, x=variable)
    plt.title(f"Distribution of {variable} \n", fontsize=20)
    plt.show()


var_quant_anal("loan_amnt", X_train)

var_quant_anal("int_rate", X_train)

var_quant_anal("annual_inc", X_train)

var_quant_anal("delinq_2yrs", X_train)

var_quant_anal("fico_range_high", X_train)

var_quant_anal("open_acc", X_train)

var_quant_anal("acc_now_delinq", X_train)

var_quant_anal("bc_util", X_train)

var_quant_anal("total_bal_ex_mort", X_train)

var_quant_anal("total_bc_limit", X_train)

# MinMax: loan_amnt, fico_range_high (skewed), int_rate (slightly skewed), bc_util
# Standard:
# Robust: Annual_inc (outliers), delinq_2yrs (extreme outliers), open_acc (outliers), acc_now_delinq (outliers),
# . total_bal_ex_mort (outliers), total_bc_limit (outliers)

# if i had more time, i woudl invest more time to explore the outliers/extreme values.

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

to_robust = [
    "annual_inc",
    "delinq_2yrs",
    "open_acc",
    "acc_now_delinq",
    "total_bal_ex_mort",
    "total_bc_limit",
]
to_minmax = ["loan_amnt", "fico_range_high", "int_rate", "bc_util"]

# using column transformer to the the step at the same time.
preprocessing = ColumnTransformer(
    transformers=[
        ("robust_scaler", RobustScaler(), to_robust),
        ("minmax_scaler", MinMaxScaler(), to_minmax),
    ],
    remainder="passthrough",
)

# take it to the data frame
X_train_scaled = preprocessing.fit_transform(X_train)

# take it back to data frame
scaled_columns = (
    to_robust
    + to_minmax
    + [col for col in X_train.columns if col not in (to_robust + to_minmax)]
)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=scaled_columns)
X_train_scaled.term.value_counts()

dummies = pd.get_dummies(X_train_scaled["term"])

X_train_scaled = pd.concat(
    [X_train_scaled, dummies.iloc[:, 0].rename("term_36m")], axis=1
)

X_train_scaled.drop(columns="term", axis=1, inplace=True)

# (t) Choose and justify a metric related to your strategy.

# (u) Train and fine-tune the chosen model (s). Examine and comment on your findings.

X_train_scaled.head()

y_train.shape

# Since the target is imbalanced, accuracy score is not the right choice, since the algorithm might easily predict
# 0 (if he predicts all 0, he gets accuracy of nearly 80%), therefore we rely here on F1 Score, Recall, and Precision.
# We choose lazy predict to indicate good base models to choice and tune. Therefore, we run the scaling things also
# on the test set.

to_robust = [
    "annual_inc",
    "delinq_2yrs",
    "open_acc",
    "acc_now_delinq",
    "total_bal_ex_mort",
    "total_bc_limit",
]
to_minmax = ["loan_amnt", "fico_range_high", "int_rate", "bc_util"]

# using column transformer to the the step at the same time.
preprocessing = ColumnTransformer(
    transformers=[
        ("robust_scaler", RobustScaler(), to_robust),
        ("minmax_scaler", MinMaxScaler(), to_minmax),
    ],
    remainder="passthrough",
)

# take it to the data frame
X_test_scaled = preprocessing.fit_transform(X_test)

# take it back to data frame
scaled_columns = (
    to_robust
    + to_minmax
    + [col for col in X_test.columns if col not in (to_robust + to_minmax)]
)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=scaled_columns)

dummies = pd.get_dummies(X_test_scaled["term"])

X_test_scaled = pd.concat(
    [X_test_scaled, dummies.iloc[:, 0].rename("term_36m")], axis=1
)

X_test_scaled.drop(columns="term", axis=1, inplace=True)
## Error we cannot use lazypredict, since it is not installed :-(
# it's a pity, since I understood this is good to identify base models to use for model optimization

from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# Crossvalidation
logistic = LogisticRegression()

skf = StratifiedKFold(n_splits=5)

cross_validate(
    logistic,
    X_train_scaled,
    y_train,
    cv=skf,
    scoring=("accuracy", "precision", "recall", "f1"),
)

# From above, we can see that the accuracy score for the logistic regression is good, and from the other scores,
# we can see that this relies on the high levels of putting into category 1.
# All other scores indicate bad fit
from sklearn.svm import SVC

svcl = SVC(gamma=0.01, kernel="poly")

# splitting train set
skf = StratifiedKFold(n_splits=5)

svcl.fit(X_train_scaled, y_train)

cross_validate(
    svcl,
    X_train_scaled,
    y_train,
    cv=skf,
    scoring=("accuracy", "precision", "recall", "f1"),
)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=-1, random_state=321)

# splitting train set
skf = StratifiedKFold(n_splits=5)

rfc.fit(X_train_scaled, y_train)

cross_validate(
    rfc,
    X_train_scaled,
    y_train,
    cv=skf,
    scoring=("accuracy", "precision", "recall", "f1"),
)

# This didn't run through in time

# We can see that logistic regression does not predict well
# however, it is recommended to use grid search or random search to optimize models
# here i choose besides logistic regression, RandomForestClassifier and SVC
# I created a param grid based on the standard parameters from the training for the grid search
# I run through each model and test each of the search algorithms
# i run tests on model estimation and save these values (accuracy, f1_score, precision, and recall)
# last I print the best model to use it further (maybe with undersampling.)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC": SVC(),
}

# Hyperparameters to be tested for each model
param_grids = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30],
    },
    "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
}

# Dictionary for storing results
results = {}


def compare_search_methods(model_name, model, param_grid):
    search_methods = {
        "GridSearchCV": GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
        ),
        "RandomizedSearchCV": RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=5,
            cv=5,
            scoring="accuracy",
            random_state=42,
        ),
    }

    results[model_name] = {}

    for search_name, search in search_methods.items():
        # Perform hyperparameter search
        search.fit(X_train_scaled, y_train)

        # Best score and hyperparameters found
        best_params = search.best_params_
        best_score = search.best_score_

        # Test on test data
        y_pred = search.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)

        # Store results
        results[model_name][search_name] = {
            "best_params": best_params,
            "best_cv_score": best_score,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_f1": test_f1,
            "test_recall": test_recall,
        }


# Run comparison for each model
for model_name, model in models.items():
    compare_search_methods(model_name, model, param_grids[model_name])

# Print results
for model_name, model_results in results.items():
    print(f"Model: {model_name}")
    for search_name, search_results in model_results.items():
        print(f"  {search_name}:")
        print(f"    Best Params: {search_results['best_params']}")
        print(f"    Best CV Score: {search_results['best_cv_score']:2f}")
        print(f"    Test Accuracy: {search_results['test_accuracy']:.2f}")
        print(f"    Test Precision: {search_results['test_precision']:.2f}")
        print(f"    Test F1: {search_results['test_f1']:.2f}")
        print(f"    Test Recall: {search_results['test_recall']:.2f}")

    print("\n")

# If i had more time, due to the imbalanced data set, I would use Undersampling to make it
# Resampling on the train base

from imblearn.under_sampling import RandomUnderSampler

undersamp = RandomUnderSampler(sampling_strategy="majority")

X_train_scaled_us, y_train_us = undersamp.fit_resample(X_train_scaled, y_train)

# run the best model with the best hyperparameters again with the undersampling method and check results
