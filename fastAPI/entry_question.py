import pandas as pd
import random
from fastapi import HTTPException

# Load dataset once at startup
file_path = "data/questions_en.xlsx"


def load_questions():
    """Loads and cleans the dataset into a DataFrame."""
    df = pd.read_excel(file_path)

    # filtering questions that do not have info on correct answer
    df = df[df["correct"].notna()]

    # making strings to lower case to make it easier to match and have less errors when checking against user-input (wrong spelled entries)
    df["use"] = df["use"].str.strip().str.lower()
    df["subject"] = df["subject"].str.strip().str.lower()

    # drop column that only has 1 valid entry
    df.drop("remark", axis=1, inplace=True)

    # fill NA's to not loose the rows due to conversion in output
    df["responseC"].fillna("", inplace=True)
    df["responseD"].fillna("", inplace=True)

    return df


# defining get questions
def insert_question(new_question: dict):
    """Adds a new row to the data set (includes a new question)."""

    # create file_path
    file_path = "data/questions_en.xlsx"

    # Loading data set of questions
    df = load_questions()

    # check for all entries
    expected_columns = [
        "question",
        "subject",
        "use",
        "correct",
        "responseA",
        "responseB",
        "responseC",
        "responseD",
        "remark",
    ]

    # create NA if column is not in dictionary
    new_row = {col: new_question.get(col, "NA") for col in expected_columns}

    # transform to dataFrame
    new_row = pd.DataFrame([new_row])

    # add the new entry
    new_df = pd.concat([df, new_row], ignore_index=True)

    # save as xlsx (as original)
    new_df.to_excel(file_path, index=False)

    return new_df.iloc[-1]
