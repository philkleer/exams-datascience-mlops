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

df = load_questions()

# creating array of valid uses 
valid_uses = df["use"].unique()
# using mapping to create dictionary of uses and valid subjects per use
valid_subjects_use = {use.lower(): df[df["use"] == use]["subject"].unique().tolist() for use in valid_uses}

def get_questions(use: str, subjects: list, num: int):
    """Filters questions based on `use` and `subjects`, then returns a random sample."""

    # creating user input all to lower to make checks easier and have less errors
    use = use.strip().lower()
    subjects = [s.strip().lower() for s in subjects]
    
    # check if num is 5, 10, or 20:
    if num not in {5, 10, 20}:
        raise HTTPException(
            status_code=400, 
            detail="Invalid entry for 'num'. Choose from [5, 10, 20]"
        )

    # check if use really exists
    if use not in valid_subjects_use:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entry for 'use'. Allowed values: {list(valid_subjects_use.keys())}"
        )

    # check if subject(s) exist(s) for use
    invalid_subjects = [s for s in subjects if s not in valid_subjects_use[use]]
    if invalid_subjects: 
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid subject(s): {invalid_subjects}. Choose from {valid_subjects_use[use]}"
        )

    # filtering df based on use and subjects
    filtered_df = df[(df["use"] == use) & (df["subject"].isin(subjects))]

    # ensuring amount of questions is available
    if num > len(filtered_df): 
        raise HTTPException(
            status_code=400, 
            detail=f"Not enough questions available. Only {len(filtered_df)} questions available."
        )

    # sampling randomly from sample amount of question set in num
    final = filtered_df.sample(n=num).to_dict(orient="records")

    return final
