from airflow import DAG
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.models import TaskInstance
from airflow.utils.state import State
from airflow.operators.python import ShortCircuitOperator
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup
from airflow.sensors.filesystem import FileSensor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
import pandas as pd
import logging
import requests
import json
import os
from joblib import dump
from io import StringIO

######################################################
#                                                    #
#             Functions without tasks                #
#                                                    #
######################################################


def create_weather_data(parent_folder, n_files=None):
    """
    Imports the data json (fetch_data()) and creates from specified variables
    a DataFrame and returns the DataFrame

    - imports the different fetched data from the API
    - loops over the fetched data and creates a DataFrame for specific variables
    - returns the df

    Arguments:
    - parent_folder (str): Path to the folder containing weather data files.
    - n_files (int, optional): Number of most recent files to process. If None, all files are used.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['temperature', 'city', 'pressure', 'date'].
    """

    # search in origin folder for files
    files = sorted(os.listdir(parent_folder), reverse=True)
    # create array of files
    if n_files:
        files = files[:n_files]

    dfs = []

    # loop through the files
    for f in files:
        with open(os.path.join(parent_folder, f), "r") as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append(
                {
                    "temperature": data_city["main"]["temp"],
                    "city": data_city["name"],
                    "pressure": data_city["main"]["pressure"],
                    "date": f.split(".")[0],
                }
            )

    df = pd.DataFrame(dfs)

    return df


def save_file(df, output_path):
    """
    Saves a DataFrame as a CSV file to the specified output path.

    - ensures existing folder before saving (creating it)
    - saves DataFrame to the given path as CSV
    - logs success messages for saving file

    Arguments:
    - df (pd.DataFrame): DataFrame to be saved
    - output_path (str): file path where the files is being saved

    Returns:
    - None
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    logging.info(f"Saved CSV: {output_path}")


def prepare_data(path_to_data="/app/clean_data/fulldata.csv"):
    """Preparing data

    This function prepares the data for modelling by creating target and features.

    - Sorts data by city and date
    - creates lagged temperature features
    - generates a target column (previous day's temperature)
    - encodes the `city`column using one-hot encoding

    Arguments:
    - path_to_data (str): file path to the CSV file containing weather data. Expects a CSV file with `city`, `date`, and `temperature` columns.

    Returns:
    - features (pd.DataFrame): DataFrame containing lagged temperature values and city dummies
    - target (pd.Series): Series representing the previous day's temperature
    """
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(["city", "date"], ascending=True)

    dfs = []

    for c in df["city"].unique():
        df_temp = df[df["city"] == c].copy()

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

    logging.info(f"Features shape: {features.shape}")
    logging.info(f"Target shape: {target.shape}")
    return features, target


def compute_model_score(model, X, y):
    """
    This function runs cross-validation on a specified model and returns its performance score.

    - uses a 3-fold cross-validation
    - evaluates performance using negative mean squared error
    - returns average negative MSE across folds

    Arguments:
    - model (sklearn.base.BaseEstimator): a scikit-learn-compatible model to be evaluated
    - X (pd.DataFrame): feature matrix for training
    - y (pd.Series): target variable

    Returns:
    - model_score (float): ean negative mean squared error (MSE) across folds. Note: The returned value is negative; a higher value is better.
    """
    # computing cross val
    cross_validation = cross_val_score(
        model, X, y, cv=3, scoring="neg_mean_squared_error"
    )

    model_score = cross_validation.mean()

    return model_score


def train_and_save_model(model, X, y, path_to_model):
    """
    This function trains a scikit-learn model on the provided data and saves it to the specified path.

    Arguments:
    - model (sklearn.base.BaseEstimator): a scikit-learn-compatible model to be evaluated
    - X (pd.DataFrame): feature matrix for training
    - y (pd.Series): target variable
    - path_to_model (str): file path where to save the model

    Returns:
    - None

    """
    # training the model
    model.fit(X, y)
    # saving model
    logging.info(f"{str(model)} saved at {path_to_model}")
    dump(model, path_to_model)


def time_check():
    first_run_timestamp_str = Variable.get("first_run_timestamp", default_var=None)

    if first_run_timestamp_str is None:
        first_run_timestamp = datetime.now()
        Variable.set(
            "first_run_timestamp", first_run_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )

        return False

    first_run_timestamp = datetime.strptime(
        first_run_timestamp_str, "%Y-%m-%d %H:%M:%S"
    )

    time_elapsed = datetime.now() - first_run_timestamp

    if time_elapsed >= timedelta(minutes=20):
        return True
    else:
        return False


######################################################
#                                                    #
#             Functions as tasks                     #
#                                                    #
######################################################


@task(
    doc="""# Fetch weather data from API
    
    This task fetches weather data from open weather map API from predefined cities and saves the data as a JSON file. Logs success or failure messages for each API request.

    - Retrieves API key and city list from Airflow variables.
    - Sends requests to OpenWeatherMap API for each city.
    - Stores the weather data in a list.
    - Extracts a timestamp from the first successful response.
    - Saves the collected data to a JSON file named after the timestamp in the `/app/raw_files/` directory.

    Arguments:
    - None

    Returns:
    - None
    """
)
def fetch_data():
    # input: airflow variables
    api_key = Variable.get("openweathermap_api_key", default_var="your_default_key")
    # get cities
    cities = Variable.get("weather_cities", default_var="recife,brasilia,natal").split(
        ","
    )

    weather_data = []
    timestamp = None

    # Loop over cities and fetch weather data
    for city in cities:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        )
        response = requests.get(url)

        if response.status_code == 200:
            city_data = response.json()
            # adding dictionary of next city to weather_data
            weather_data.append(city_data)

            # extract timestamp
            if timestamp is None:
                timestamp = city_data["dt"]

                logging.info(f"Data for {city} received!")
        else:
            logging.info(
                f"Error fetching data for {city}: {response.status_code}, {response.text}"
            )

    # test directory
    output_dir = "/app/raw_files/"
    # create if not already there
    os.makedirs(output_dir, exist_ok=True)

    # convert timestamp to str
    if timestamp:
        formatted_time = datetime.fromtimestamp(timestamp, UTC).strftime(
            "%Y-%m-%d %H:%M"
        )

        # create filename
        filename = os.path.join(output_dir, f"{formatted_time}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(weather_data, f, ensure_ascii=False, indent=4)

        logging.info(
            f"The data of the three cities has been saved to {filename}! :-) (Task 1)"
        )
    else:
        logging.info("No valid data retrieved. File was not created :-(. (Task 1)")


# verifications
@task(
    doc=""" # File Verification

    Checks if the specified file exists before proceeding with downstream tasks.

    - Uses FileSensor to continuously check for the presence of the file.
    - Waits for a maximum of `timeout` seconds, checking every `poke_interval` seconds.
    - logs info about success or failure

    Arguments:
    - name_task (str): name of the task (used as `task_id` in Airflow)
    - filepath (str): path to the file to be verified

    Returns:
    - bool: True if the file is detected within the timeout period.
"""
)
def verify_file_exists(name_task, filepath):
    # here I add initially some issues therefore, I have more logging.
    logger = logging.getLogger(name_task)

    logger.info(f"Starting to check if file {filepath} exists.")

    file_sensor = FileSensor(
        task_id=name_task,
        filepath=filepath,
        poke_interval=60,
        timeout=600,
        mode="reschedule",
    )

    try:
        file_sensor.execute(context={})  # Airflow will handle retries internally
        logger.info(f"File {filepath} detected successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to detect file {filepath}: {str(e)}")
        return False


@task(
    doc="""# Save latest weather data
    This task imports the weather data and selects the latest 20 files to create a data set of these. It saves the data file in `/app/clean_data/data.csv`.
    
    Arguments:
    - None

    Returns:
    - None
    """
)
def latest_weather_data():
    df = create_weather_data("/app/raw_files", n_files=20)
    save_file(df, "/app/clean_data/data.csv")
    logging.info("Saved dataset as `/app/clean_data/data.csv` (Task 2)!")


@task(
    doc="""# Save full weather data
    
    This task imports the weather data and creates a data set of the full data. It saves the data file in `/app/clean_data/fulldata.csv`.
"""
)
def full_weather_data():
    df = create_weather_data("/app/raw_files")
    save_file(df, "/app/clean_data/fulldata.csv")
    logging.info("Saved dataset as `/app/clean_data/fulldata.csv` (Task 3)!")


@task(
    doc=""" # Loads and prepares weather data

    This task uses the function `prepare_data()` to create feature matrix and target for the modelling.

    - Reads a CSV file containing weather data.
    - Calls `prepare_data()` to generate lagged temperature features and a target variable.
    - Returns the processed feature matrix (`X`) and target variable (`y`).

    Arguments:
    - path (str): file path to the CSV file containing weather data. Expects a CSV file with `city`, `date`, and `temperature` columns.

    Returns:
    - X (pd.DataFrame): Feature matrix containing lagged temperature values and one-hot encoded city variables.
    - y (pd.Series): Target variable representing the previous day's temperature.
"""
)
def load_data(path="/app/clean_data/fulldata.csv"):
    X, y = prepare_data(path)

    # making X and y serializable (CSV strings)
    csv_X = X.to_csv(index=False)
    csv_y = y.to_csv(index=False)

    return {"X": csv_X, "y": csv_y}


@task(
    doc=""" # Evaluates and saves scores

    This task compute the model (`compute_model_score()`) and saves the score with xcom. 

    - reformat the serialized data from xcom
    - train the model in cross-validation with 3-folds (`compute_model_score()`)
    - gets the model name
    - pushes the score in a dictionary with model name with xcom
    - logs the validation score
"""
)
def evaluate_model(model, Xcom_data):  # , ti, X, y
    # load data
    X = pd.read_csv(StringIO(Xcom_data["X"]))
    y = pd.read_csv(StringIO(Xcom_data["y"])).squeeze()

    score = compute_model_score(model, X, y)
    model_name = model.__class__.__name__
    # ti.xcom_ush(key=model_name, value=score)
    logging.info(f"{model_name} cross-validation score: {score}")
    return score


@task(
    doc=""" # Comparing scores
    This task compares the model performance scores and selects the best model based on the highest score.

    - Retrieves the scores for each model from XComs.
    - Identifies the best model by selecting the one with the highest (least negative) score.
    - Retrains the best model using the full dataset without cross-validation.
    - Saves the best model to disk.
    - Logs success and the name of the best model.

    Arguments:
    - X (pd.DataFrame): Feature matrix used for model training.
    - y (pd.Series): Target variable used for model training.
    - ti (TaskInstance): Airflow TaskInstance used to retrieve XComs.
    
    Returns:
    - None

"""
)
def select_best_model(Xcom_data, task4a, task4b, task4c):  # , ti
    scores = {
        "LinearRegression": task4a,
        "DecisionTreeRegressor": task4b,
        "RandomForestRegressor": task4c,
    }

    best_model_name = max(scores, key=scores.get)
    best_model = {
        "LinearRegression": LinearRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
    }

    # load data
    X = pd.read_csv(StringIO(Xcom_data["X"]))
    y = pd.read_csv(StringIO(Xcom_data["y"])).squeeze()
    train_and_save_model(
        best_model[best_model_name](), X, y, "/app/clean_data/best_model.pickle"
    )
    logging.info(f"Best model: {best_model_name}. Data was retrained and saved!")


######################################################
#                                                    #
#             DAG settings & tasks                   #
#                                                    #
######################################################

# Default arguments for DAG tasks
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 12),  # Static start date
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


# Define the DAG
@dag(
    dag_id="weather_pipeline",
    default_args=default_args,
    schedule_interval="*/1 * * * *",  # Runs every minute
    catchup=False,
    tags=["weather_pipeline", "airflow_exam"],
)
def weather_pipeline():
    # Task 1: fetching data
    task1 = fetch_data()

    # Time checking to fetch enough data
    # waiting_time = time_check()

    # Circuit Operator
    check_time = ShortCircuitOperator(
        task_id="branch_check", python_callable=time_check, provide_context=True
    )

    # Task 2: creating data sets
    task2 = latest_weather_data()
    task3 = full_weather_data()

    # verification step
    with TaskGroup(
        "verification_group", tooltip="Verify file existence"
    ) as verification_group:
        verify_task2 = verify_file_exists("task2", "/app/clean_data/data.csv")
        verify_task3 = verify_file_exists("task3", "/app/clean_data/fulldata.csv")

    # preliminary step: load_data
    load_task4 = load_data("/app/clean_data/fulldata.csv")
    # Task 4: Task group with three different models
    with TaskGroup(
        "model_evaluation", tooltip="Evaluate different models"
    ) as model_group:
        task4a = evaluate_model(LinearRegression(), load_task4)
        task4b = evaluate_model(DecisionTreeRegressor(), load_task4)
        task4c = evaluate_model(RandomForestRegressor(), load_task4)

    # Task 5: creating full model
    task5 = select_best_model(load_task4, task4a, task4b, task4c)

    # Setting task dependencies
    task1 >> check_time >> [task2, task3] >> verification_group
    [verification_group] >> load_task4 >> model_group
    model_group >> task5


# Initiliazing
weather_pipeline_dag = weather_pipeline()
