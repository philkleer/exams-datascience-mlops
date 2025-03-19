import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model, X, y, cv=3, scoring="neg_mean_squared_error"
    )

    model_score = cross_validation.mean()

    return model_score


def train_and_save_model(model, X, y, path_to_model="./model/best_model.pckl"):
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), "saved at ", path_to_model)
    dump(model, path_to_model)


def prepare_data(path_to_data="./clean_data/fulldata.csv"):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(["city", "date"], ascending=True)

    dfs = []

    for c in df["city"].unique():
        df_temp = df[df["city"] == c]

        # creating target (previous temperature)
        df_temp.loc[:, "target"] = df_temp["temperature"].shift(1)

        # creating features lag vars
        for i in range(1, 10):
            df_temp.loc[:, "temp_m-{i}"] = df_temp["temperature"].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(dfs, axis=0, ignore_index=False)

    # deleting date variable
    df_final = df_final.drop(["date"], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    features = df_final.drop(["target"], axis=1)
    target = df_final["target"]

    return features, target


def evaluate_model(model, X, y, ti):
    score = compute_model_score(model, X, y)
    model_name = model.__name__
    ti.xcom_ush(key=model_name, value=score)
    print(f"{model_name} cross-validation score: {score}")


def load_data(path="/app/clean_data/fulldata.csv"):
    X, y = prepare_data(path)
    return X, y


def select_best_model(X, y, ti):
    scores = {
        "LinearRegression": ti.xcom_pull(task_ids="train_lr", key="LinearRegression"),
        "DecisionTreeRegressor": ti.xcom_pull(
            task_ids="train_dt", key="DecisionTreeRegressor"
        ),
        "RandomForestRegressor": ti.xcom_pull(
            task_ids="train_rf", key="RandomForestRegressor"
        ),
    }

    best_model_name = min(scores, key=scores.get)
    best_model = {
        "LinearRegression": LinearRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
    }

    train_and_save_model(
        best_model[best_model_name], X, y, "/app/model/best_model.pickle"
    )
    print(f"Best model: {best_model_name}. Data was retrained and saved!")


#


# TEsts
def prepare_data(path_to_data="./clean_data/fulldata.csv"):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(["city", "date"], ascending=True)

    dfs = []

    for c in df["city"].unique():
        df_temp = df[df["city"] == c]

        # creating target (previous temperature)
        df_temp.loc[:, "target"] = df_temp["temperature"].shift(1)

        # creating features lag vars
        for i in range(1, 10):
            df_temp.loc[:, f"temp_m-{i}"] = df_temp["temperature"].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(dfs, axis=0, ignore_index=False)

    # deleting date variable
    df_final = df_final.drop(["date"], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    features = df_final.drop(["target"], axis=1)
    target = df_final["target"]

    return features, target


features, target = prepare_data(
    path_to_data="./sprint3/exam_airflow/clean_data/fulldata.csv"
)

features.shape
target.shape

df = pd.read_csv("./sprint3/exam_airflow/clean_data/fulldata.csv")
# ordering data according to city and date
df = df.sort_values(["city", "date"], ascending=True)

dfs = []

for c in df["city"].unique():
    df_temp = df[df["city"] == c]

    # creating target (previous temperature)
    df_temp.loc[:, "target"] = df_temp["temperature"].shift(1)

    # creating features lag vars
    for i in range(1, 10):
        df_temp.loc[:, f"temp_m-{i}"] = df_temp["temperature"].shift(-i)

    # deleting null values
    df_temp = df_temp.dropna()

    dfs.append(df_temp)

# concatenating datasets
df_final = pd.concat(dfs, axis=0, ignore_index=False)
