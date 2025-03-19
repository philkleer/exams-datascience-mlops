# Exam FastAPI

In this compressed folder you find the following files: 

- `exploration.py`: file for exploration of the data set
- folder `data`: folder with the file `questions_en.xlsx`
- `main.py`: main file for the API
- `authentication.py`: subfile for the authentication process
- `load_questions.py`: subfile for loading questions
- `entry_question.py`: subfile for adding a question with admin access
- `curls.md`: curls for each case (not working and working) with output-solutions

For the part on authentication, I was quite unsure after the module and used other literature and information to create a solution. Probably, I haven't had in mind a possible simpler solution form the module; however, the solution works fine. 

I had some troubles with the NA's, since I chose to create a dict in JSON style as output again, since I thought that should be the aim (since it is expected from APIs). However, after finding that `.to_dict()` can't handle NA's, it was quite easy to overcome. 