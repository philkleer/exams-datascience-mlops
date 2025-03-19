############## SETTING UP API  ####################
from fastapi import FastAPI, Depends, Query, HTTPException

# loading authentication parts
from authentication import authenticate_user, authenticate_admin

# loading the question
from load_questions import get_questions

# loadig module to add question
from entry_question import insert_question

# for the model
from pydantic import BaseModel
from typing import Optional

# Creating Api
app = FastAPI()


# defining root
@app.get("/")
def read_root():
    return {"message": "FastAPI MCQ API is running!"}


# defining get questions
@app.get("/mcq")
def get_mcqs(
    use: str = Query(..., description="Type of test"),
    subjects: str = Query(..., description="String of subjects"),
    num: int = Query(5, enum=[5, 10, 20], description="Number of MCQs"),
    user: str = Depends(authenticate_user),
):
    """Fetches a random set of MCQs based on selected criteria. Checks if combination is available!"""

    try:
        # split input string (easier to write than list) into list of subjects
        subjects = subjects.split(",")

        questions = get_questions(use, subjects, num)

        return {"user": user, "questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# defining a model to test against
class QuestionModel(BaseModel):
    question: str
    subject: str
    use: str
    correct: str
    responseA: str
    responseB: str
    # there need at least two choices for responses
    responseC: Optional[str] = "NA"
    responseD: Optional[str] = "NA"
    remark: str


# defining the admin api to add a question
@app.post("/admin")
def admin_question(
    question: QuestionModel,
    user: str = Depends(authenticate_admin),
):
    """Add a new question to the dataset if admin rights are given!"""

    try:
        new_row = insert_question(question.dict())
        line = new_row
        response = {
            "message": "Question added successfully and DataFrame saved!",
            "new_row": line,
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
