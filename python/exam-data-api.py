# Introduction to Data APIs
# Exam
# In this exam, your mission will be to build KPIs around data from a Data Science training organization.

# The problem with this organization is that it wants to analyze the performance of its students with regard to the courses it offers.

# The organization offers a catalog of 6 courses.
# A total of 100 learners received training.
# Learners are divided into 3 courses: Data Analyst, Scientist and Engineer.
# Each learner chooses 4 courses from the catalog for their training.
# Each course is delivered in 10 face-to-face sessions. At the end of these sessions, learners must pass an exam marked out of 20.
# To assess the relevance of its training catalogue, the organization would like to have the following KPIs for each course:

# Number of registered students
# Cursus most represented among the students of this course
# Attendance rate of students at course sessions
# Exam participation rate
# Average marks
# Exam pass rate
# The data used to calculate these KPIs is distributed in 4 different endpoints:

# "https://examen-api.s3.eu-west-1.amazonaws.com/Students" returns the list of identifiers of learners who have completed training.

# "https://examen-api.s3.eu-west-1.amazonaws.com/Student/{ID}" returns the registration data relating to a learner. {ID} corresponds to the identifier of a learner.

# Here is an example of the data returned by this endpoint:

# {
#     "StudentID": 0,
#     "StudentName": "John Chambers",
#     "StudentCursus": "DS",
#     "StudentCourses": [
#         3,
#         2,
#         4,
#         1
#     ]
# }
# Note that the "StudentCourses" key contains the list of identifiers of the courses in which the learner is registered.

# "https://examen-api.s3.eu-west-1.amazonaws.com/Attendance/{ID}" returns the attendance data recorded for a learner in face-to-face sessions. {ID} corresponds to the identifier of a learner.
# Here is an example of the data returned by this endpoint:

# {
#     "StudentID": 0,
#     "StudentAttendance": [
#         {
#             "2": "17-10-2022"
#         },
#         {
#             "3": "19-09-2022"
#         },
#         {
#             "1": "07-11-2022"
#         },
#     ]
# }
# This time, we will notice that the list of presences noted is stored in a list of dictionaries. Each dictionary is in the form {"Course ID": Session Date}.

# "https://examen-api.s3.eu-west-1.amazonaws.com/Grades/{ID}" returns data of the grades obtained by a learner to the courses where he is registered. {ID} corresponds to the identifier of a learner.
# Here is an example of the data returned by this endpoint:

# {
#     "StudentID": 2,
#     "StudentGrades": [
#         {
#             "3": 0.0
#         },
#         {
#             "4": 16.0
#         },
#         {
#             "5": 11.5
#         },
#         {
#             "0": 7.2
#         }
#     ]
# }
# As for the previous endpoint, grades are stored in a dictionary list where each dictionary is in the form {"Course ID": Grade obtained}.

# I. Extract data
# We will start by retrieving the list of learner identifiers. This list will be useful to us for the rest of the examination.

# (a) Using a GET request on the "https://examen-api.s3.eu-west-1.amazonaws.com/Students", retrieve the list of student IDs and store the result in a list named student_list.
import requests

response = requests.get('https://examen-api.s3.eu-west-1.amazonaws.com/Students')

response = response.json()

student_list = response['StudentList']

# (b) Define a function named extract_enrollments which will take as argument a list of learner IDs and return the following DataFrame:
# CourseID	StudentID	StudentName	StudentCursus
# 0	3	0	John Chambers	DS
# 1	2	0	John Chambers	DS
# 2	4	0	John Chambers	DS
# 3	1	0	John Chambers	DS
# 4	4	1	David Williams	DA
# ...	...	...	...	...
# 395	5	98	Kevin Cortinas	DE
# 396	5	99	Jennifer Reighard	DE
# 397	2	99	Jennifer Reighard	DE
# 398	0	99	Jennifer Reighard	DE
# 399	3	99	Jennifer Reighard	DE
# The endpoint to use is "https://examen-api.s3.eu-west-1.amazonaws.com/Student/{ID}" where {ID} is the identifier of a learner.

# Tips:

# Don't define the function right away. First develop a suite of statements to handle a single learner, then generalize your approach into a loop that loops through a list of ids, then finally define the function.
# The easiest way to do this is to instantiate a DataFrame from a list of dictionaries in the following form:
# {
#   'CourseID': id of a course,
#   'StudentID': learner identifier,
#   'StudentName': name of the learner,
#   'StudentCursus': learner's curriculum
# }
# For each student, you will have to produce 4 dictionaries of this form. The final list should therefore contain 400 elements.

import requests
import pandas as pd


# single approach
learner_id = 0

source = f"https://examen-api.s3.eu-west-1.amazonaws.com/Student/{learner_id}"
response = requests.get(source)

response = response.json()

response


courses = response["StudentCourses"]

rows = [
    {
        "StudentCourse": course, 
        "StudentID": response["StudentID"],
        "StudentName": response["StudentName"], 
        "StudentCursus": response["StudentCursus"]
    }
    for course in courses
]

df = pd.DataFrame(rows)
print(df)


def extract_enrollments(learner_list): 
    all_rows = []
    
    for learner_id in learner_list:
        source = f"https://examen-api.s3.eu-west-1.amazonaws.com/Student/{learner_id}"
        response = requests.get(source)
        response = response.json()
        courses = response["StudentCourses"]
        
        rows = [
            {
                "StudentCourse": course, 
                "StudentID": response["StudentID"],
                "StudentName": response["StudentName"], 
                "StudentCursus": response["StudentCursus"]
            }
            for course in courses
        ]
        all_rows.extend(rows)
    
    return pd.DataFrame(all_rows)

# (c) Execute the function extract_enrollments with the list student_list as an argument and store the result produced in a DataFrame named enrollments.
# Insert your code here
enrollments = extract_enrollments(student_list)
    
enrollments.head()

# (d) Define a function named extract_attendances which will take as argument a list of learner IDs and return the following DataFrame:
# CourseID	StudentID	Date
# 0	2	0	17-10-2022
# 1	3	0	19-09-2022
# 2	2	0	10-10-2022
# 3	2	0	12-09-2022
# 4	3	0	07-11-2022
# ...	...	...	...
# 2954	0	99	26-09-2022
# 2955	0	99	19-09-2022
# 2956	2	99	07-11-2022
# 2957	3	99	26-09-2022
# 2958	3	99	31-10-2022
# The endpoint to use is "https://examen-api.s3.eu-west-1.amazonaws.com/Attendance/{ID}" where {ID} is the identifier of a learner.

# Tips :

# If we assume that each dictionary indicating attendance at a session has the form event = {Course ID: Date}, then we can retrieve the course identifier by doing list(event.keys ())[0] and the date with list(event.values())[0].

# To avoid join problems later, don't forget to make sure that the "CourseID" column is of type integer.

learner_id = 0

source = f"https://examen-api.s3.eu-west-1.amazonaws.com/Attendance/{learner_id}"
response = requests.get(source)

response = response.json()

response


for event in response['StudentAttendance']:
    course_id = int(list(event.keys())[0])
    date = list(event.values()) 
    
    rows = [
    {
        "CourseID": course_id, 
        "StudentID": learner_id,
        "Date": date
    }
]
    
df = pd.DataFrame(rows)

df['CourseID'] = df["CourseID"].astype(int)

df.head()

def extract_enrollments(learner_list): 
    all_rows = []
    
    for learner_id in learner_list:
        source = f"https://examen-api.s3.eu-west-1.amazonaws.com/Attendance/{learner_id}"
        response = requests.get(source)
        response = response.json()
        attendances = response['StudentAttendance']
        
        for event in attendances:
            course_id = int(list(event.keys())[0])
            date = next(iter(event.values()))
            
            rows = [
                {
                    "CourseID": course_id, 
                    "StudentID": learner_id,
                    "Date": date
                }
            ]
            
        all_rows.extend(rows)        
    
    return pd.DataFrame(all_rows)

# (e) Execute the function extract_attendances with the list student_list as an argument and store the result produced in a DataFrame named attendances.
attendances = extract_enrollments(student_list)
attendances.info()
attendances.head()

# (f) Define a function named extract_grades which will take as argument a list of learner IDs and return the following DataFrame:
# CourseID	StudentID	Grade	Attended	Success
# 0	3	0	8.6	True	False
# 1	2	0	15.4	True	True
# 2	4	0	10.2	True	True
# 3	1	0	8.2	True	False
# 4	4	1	0	False	False
# ...	...	...	...	...	...
# 395	5	98	10.2	True	True
# 396	5	99	10.5	True	True
# 397	2	99	13	True	True
# 398	0	99	8.5	True	False
# 399	3	99	10.9	True	True
# The endpoint to use is "https://examen-api.s3.eu-west-1.amazonaws.com/Grades/{ID}" where {ID} is the identifier of a learner.

learner_id = 0

source = f"https://examen-api.s3.eu-west-1.amazonaws.com/Grades/{learner_id}"
response = requests.get(source)

response = response.json()

response


for event in response['StudentGrades']:
    course_id = int(list(event.keys())[0])
    grade = next(iter(event.values()))
    
    rows = [
    {
        "CourseID": course_id, 
        "StudentID": learner_id,
        "Grade": grade
    }
]
    
df = pd.DataFrame(rows)

df['CourseID'] = df["CourseID"].astype(int)
df['Grade'] = df['Grade'].astype(float)

df['Attended'] = df['Grade'] > 0

# assuming success from 50% on
df['Success'] = df['Grade'] >= 10


df.info()

response

def extract_grades(learner_list): 
    all_rows = []
    
    for learner_id in learner_list:
        source = f"https://examen-api.s3.eu-west-1.amazonaws.com/Grades/{learner_id}"
        response = requests.get(source)
        response = response.json()
        grades = response['StudentGrades']
        
        for event in grades:
            course_id = int(list(event.keys())[0])
            grade = next(iter(event.values()))
            
            rows = [
                {
                    "CourseID": course_id, 
                    "StudentID": learner_id,
                    "Grade": grade
                }
            ]
            
        all_rows.extend(rows)
        
        df = pd.DataFrame(all_rows)
        df['CourseID'] = df['CourseID'].astype(int)
        df['Grade'] = df['Grade'].astype(float)
        df['Attended'] = df['Grade'] > 0
        df['Success'] = df['Grade'] >= 10
    
    return df

# (g) Execute the function extract_grades with the list student_list as an argument and store the result produced in a DataFrame named grades.
grades = extract_grades(student_list)

grades.head()

# II. Calculation of KPIs
# (a) Define a function named transform_enrollments which will take the DataFrame enrollments as an argument and calculate the following KPIs:
# Number of enrolled learners in a column named "EnrolledStudents".
# Most frequent cursus among learners in a column named "MajorityCursus".
# Hint: You can rename the columns of a DataFrame using the following method:

# df = df.rename(columns = {
#         "Column name" : "New name",
#         ...
#         })
# (b) Test the transform_enrollments function. We should get the following DataFrame:
# CourseID	EnrolledStudents	MajorityCursus
# 0	0	66	DS
# 1	1	62	DA
# 2	2	66	DA
# 3	3	60	DA
# 4	4	71	DA
# 5	5	75	DS

def transform_enrollments(enrollments):
    unique_students = enrollments.groupby('StudentCourse').size().reset_index(name='EnrolledStudents')
    
    mode_course = enrollments.groupby('StudentCourse')['StudentCursus'].agg(lambda x: x.mode()[0]).reset_index(name='MajorityCursus')
    
    result_df = pd.merge(unique_students, mode_course, on='StudentCourse')
    
    result_df = result_df.rename(
        columns = {
                'StudentCourse': 'CourseID'
        }
    )
    return result_df
transform_enrollments(enrollments)

# (c) Define a function named transform_attendances which will take the DataFrame attendances as an argument and calculate the attendance rate at the sessions of each course. We will store the result in a column named "AttendanceRate".

# (d) Testing the transform_attendances function, we should get the following DataFrame:

# CourseID	AttendanceRate
# 0	0	0.710606
# 1	1	0.822581
# 2	2	0.759091
# 3	3	0.681667
# 4	4	0.726761
# 5	5	0.738667
# Hints:

# Each course is broken down into 10 sessions. To calculate the attendance rate for a course, simply count the number of sessions where each learner was present per course and then divide it by 10.

# In the groupby method, it is possible to group individuals according to several variables. For example attendances.groupby(["StudentID", "CourseID"]) to group by learner and then by course.

# Consider using the reset_index method to remove columns used for a groupby operation from the index.

def transform_attendances(attendances):  
    attended_sessions = attendances.groupby(['StudentID', 'CourseID'])['Date'].nunique().reset_index(name='SessionsAttended')
    
    total_sessions = attended_sessions.groupby('CourseID')['SessionsAttended'].sum().reset_index(name='TotalSessions')
    
    total_students = attended_sessions.groupby('CourseID')['StudentID'].count().reset_index(name='TotalStudents')
    
    total_sessions['AttendanceRate'] = total_sessions['TotalSessions'] / 10
    
    result_df = total_sessions[['CourseID', 'AttendanceRate']]
    
    return result_df
transform_attendances(attendances)

# (e) Define a function named transform_grades which will take the DataFrame grades as an argument and calculate the following KPIs:

# Exam attendance rate in a column named "ExamAttendanceRate". It will be assumed that a learner was present for the exam if his mark is strictly greater than 0.
# Exam pass rate in a column named "ExamSuccessRate". A learner will be assumed to have passed the exam if their score is greater than or equal to 10.
# Average of the marks in a column named "ExamAverage". Only grades strictly above 0 will be counted in this average.
# (f) Test the transform_grades function. We should get the following DataFrame:

# CourseID	ExamAttendanceRate	ExamSuccessRate	ExamAverage
# 0	0	0.909091	0.378788	9.27833
# 1	1	0.919355	0.354839	9.02105
# 2	2	0.909091	0.787879	13.4067
# 3	3	0.9	0.583333	11.213
# 4	4	0.873239	0.43662	10.2565
# 5	5	0.88	0.52	10.5742
# Hints:

# We will start by defining in grades a column indicating whether the learner was present for the exam and a column indicating whether he passed the exam.

# To calculate the average for the exam, for each course, we can first calculate the sum of the marks then divide this sum by the number of students present at the exam.

def transform_grade(grades):
    grouped = grades.groupby('CourseID').agg(
        ExamAttendanceRate=('Attended', 'mean'),
        ExamSuccessRate=('Success', 'mean'),
        TotalGrades=('Grade', 'sum'),
        TotalAttendees=('Attended', 'sum')
    ).reset_index()
    
    grouped['ExamAverage'] = grouped['TotalGrades'] / grouped['TotalAttendees']
    
    grouped = grouped.drop(columns=['TotalGrades', 'TotalAttendees'])
    
    return grouped
transform_grade(grades)

# (g) Using the functions defined previously, perform a merge to obtain the following DataFrame:
# CourseID	EnrolledStudents	MajorityCursus	AttendanceRate	ExamAttendanceRate	ExamSuccessRate	ExamAverage
# 0	0	66	DS	0.710606	0.909091	0.378788	9.27833
# 1	1	62	DA	0.822581	0.919355	0.354839	9.02105
# 2	2	66	DA	0.759091	0.909091	0.787879	13.4067
# 3	3	60	DA	0.681667	0.9	0.583333	11.213
# 4	4	71	DA	0.726761	0.873239	0.43662	10.2565
# 5	5	75	DS	0.738667	0.88	0.52	10.5742	ç
# (h) In your opinion, which exams are too difficult? Which exams are too easy?

df1 = transform_enrollments(enrollments)
df2 = transform_attendances(attendances)
df3 = transform_grade(grades)

final_df = df1.merge(df2.merge(df3, on='CourseID'), on='CourseID')
final_df.head()

