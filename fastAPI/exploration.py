# Evaluation task
from fastapi import HTTPException

# Data is saved under data/questions_en.xlsx

############## DATA EXPLORATION ####################
# libraries
import pandas as pd

# loading data
location = "data/questions_en.xlsx"
questions = pd.read_excel(location)

# get-to-know with data
print(questions.info())
# 9 features, 76 rows
# response D is not always given, correct is only given for 68 out of 76
# remark is nearly never given

# show first rows
print(questions.head())

# entry 68 to 75 have no indication for correct answers
print(questions.correct.tail(n=10))

# checking unique values for subject and use
questions.subject.unique()
questions.use.unique()
questions.remark

# Necessities for the API:
# - remove remark, it has only 1 valid value and NA values generate error when turning into JSON
# - handling of NA's in answerC and answerD, filling with empty string
# - filter for questions that do not have a value in `correct`
# - check that numbers of question can only be 5, 10, or 20
# - check if combination of subject and use is within valid options, if not error message
