import os
import pandas as pd
import numpy as np
import json
import logging


# i think having the loading only once is more efficient
def create_weather_data(parent_folder, n_files=None):
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
                    "pression": data_city["main"]["pressure"],
                    "date": f.split(".")[0],
                }
            )

    df = pd.DataFrame(dfs)

    return df


def save_file(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # create df
    df.to_csv(output_path, index=False)

    logging.info(f"Saved CSV: {output_path}")


# task2
def process_latest_weather_data():
    df = create_weather_data("./sprint3/exam_airflow/raw_files", n_files=20)
    save_file(df, "./sprint3/exam_airflow/clean_data/data.csv")


# task3
def process_full_weather_data():
    df = create_weather_data("./sprint3/exam_airflow/raw_files")
    save_file(df, "./sprint3/exam_airflow/clean_data/fulldata.csv")


process_latest_weather_data()

process_full_weather_data()
