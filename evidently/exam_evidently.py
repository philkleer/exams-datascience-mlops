import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json
import os

from sklearn import datasets, model_selection
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import anderson_ksamp

from evidently.metrics import (
    RegressionQualityMetric,
    RegressionErrorPlot,
    RegressionErrorDistribution,
)
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

# ignore warnings
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# create a report folder
report_folder = "reports"
os.makedirs(report_folder, exist_ok=True)


# custom functions
def _fetch_data() -> pd.DataFrame:
    content = requests.get(
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        verify=False,
    ).content

    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(
            arc.open("hour.csv"), header=0, sep=",", parse_dates=["dteday"]
        )

    return raw_data


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)),
        axis=1,
    )

    return raw_data


# Model training & feature selection
def train_model(
    data: pd.DataFrame,
    numerical_features: list,
    categorical_features: list,
    target: str,
) -> pd.DataFrame:
    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data[numerical_features + categorical_features], data[target], test_size=0.3
    )

    # Model training
    regressor = RandomForestRegressor(random_state=0, n_estimators=50)
    regressor.fit(X_train, y_train)

    # Predictions
    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)

    # creating reference data
    trained_data = X_train.copy()
    trained_data["cnt"] = y_train.values
    trained_data["prediction"] = preds_train

    # creating current data
    tested_data = X_test.copy()
    tested_data["cnt"] = y_test.values
    tested_data["prediction"] = preds_test

    return regressor, trained_data, tested_data


# Add actual target and prediction columns to the training data for later performance analysis
def generate_report(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, column_mapping, metrics
):
    # initiliaze the report
    report = Report(metrics=metrics)

    # sorting if reference_data is existing
    if reference_data is not None:
        reference_data = reference_data.sort_index()

    # running the report
    report.run(
        reference_data=reference_data,
        current_data=current_data.sort_index(),
        column_mapping=column_mapping,
    )

    # returning the report
    return report


def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    This function will be useful to you
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")


def main():
    print("Starting script ...", end="\r")

    # Define constants for workspace and project details
    WORKSPACE_NAME = "exam_evidently"
    workspace = Workspace.create(WORKSPACE_NAME)

    # create variables
    target = "cnt"
    prediction = "prediction"
    num_feats = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
    cat_feats = ["season", "holiday", "workingday"]

    # column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = num_feats
    column_mapping.categorical_features = cat_feats

    # loading data (Step 1)
    print("Loading data ...", end="\r")
    raw_data = _process_data(_fetch_data())
    print("Loaded data ✅")
    # creating subsets
    # Reference and current data split
    reference_jan11 = raw_data.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"]

    # training model (Step 2)
    regressor, trained_data, tested_data = train_model(
        reference_jan11, num_feats, cat_feats, target
    )

    print("Creating model validation report ...", end="\r")
    # report1 = generate_report(trained_data, tested_data)
    report1 = generate_report(
        trained_data, tested_data, column_mapping, [RegressionPreset()]
    )
    # save model validation report
    PROJECT_NAME = "evidently_exam_task1"
    PROJECT_DESCRIPTION = "Evidently Exam DataScientest.com"
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report1)

    # report1.save_html('model_validation_report.html')
    print("Created model validation report ✅")

    # run complete model for january (production model, Step 3)
    print("Creating model validation report ...", end="\r")

    # re-ftting model and calculting predictions on whole january
    regressor.fit(reference_jan11[num_feats + cat_feats], reference_jan11[target])
    reference_jan11["prediction"] = regressor.predict(
        reference_jan11[num_feats + cat_feats]
    )

    report2 = generate_report(
        reference_data=None,
        current_data=reference_jan11,
        column_mapping=column_mapping,
        metrics=[RegressionPreset()],
    )
    # save model drift report
    # report2.save_html('production_model_drift_report.html')
    PROJECT_NAME = "evidently_exam_task2"
    PROJECT_DESCRIPTION = "Evidently Exam DataScientest.com"
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report2)
    print("Created production model drift report ✅")

    # Step 4
    print("Creating model drift report for february weeks ...", end="\r")
    # creating report for each week in february:
    weeks = [
        ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
        ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
        ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
    ]

    # creating dictionary to gather drift scores
    week_scores = {}

    # looping through the weeks to get each report
    for i, (start, end) in enumerate(weeks, 1):
        # subsetting data
        week_data = raw_data.loc[start:end]

        # creating prediciton
        week_data["prediction"] = regressor.predict(week_data[num_feats + cat_feats])

        # creating and saving report
        report3 = generate_report(
            reference_jan11,
            week_data,
            column_mapping=column_mapping,
            metrics=[RegressionPreset()],
        )
        # report3.save_html(os.path.join(report_folder, f'week_{i}_model_drift_report.html'))
        PROJECT_NAME = "evidently_exam_task3"
        PROJECT_DESCRIPTION = "Evidently Exam DataScientest.com"
        add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report3)

        # saving drift score
        mean_abs_error = report3.as_dict()["metrics"][0]["result"]["current"][
            "mean_abs_error"
        ]
        week_scores[i] = mean_abs_error

    print("Created model drift report for february weeks ✅")

    # intermediate step: getting worst week (highest drift score)
    print("Creating worst week drift report ...", end="\r")
    worst_week_index = max(week_scores, key=week_scores.get) - 1

    worst_week = weeks[worst_week_index]

    worst_week_df = raw_data.loc[worst_week[0] : worst_week[1]]

    worst_week_df["prediction"] = regressor.predict(
        worst_week_df[num_feats + cat_feats]
    )

    # Step 5
    # Target Drift Report on the worst week
    report_worst_target = generate_report(
        reference_jan11, worst_week_df, column_mapping, [TargetDriftPreset()]
    )

    # report_worst_target.save_html('target_drift_report.html')
    PROJECT_NAME = "evidently_exam_task4"
    PROJECT_DESCRIPTION = "Evidently Exam DataScientest.com"
    add_report_to_workspace(
        workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report_worst_target
    )
    print("Created worst week drift report ✅")
    # Step 6
    # Analyse Data drift
    print("Creating data drift last week only num ...", end="\r")

    # dropping all categorical features from the column mapping
    column_mapping.categorical_features = []

    # only for the last week and only numerical variables
    last_week_data = raw_data.loc[weeks[2][0] : weeks[2][1]]
    last_week_data["prediction"] = regressor.predict(
        last_week_data[num_feats + cat_feats]
    )

    # re-initaite Column mapping
    report_last_num = generate_report(
        reference_jan11,
        last_week_data,
        column_mapping=column_mapping,
        metrics=[DataDriftPreset()],
    )

    PROJECT_NAME = "evidently_exam_task5"
    PROJECT_DESCRIPTION = "Evidently Exam DataScientest.com"
    add_report_to_workspace(
        workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report_last_num
    )
    # report_last_num.save_html('data_drift_report_last_week_only_num.html')
    print("Created data drift last week only num ✅")


if __name__ == "__main__":
    main()
