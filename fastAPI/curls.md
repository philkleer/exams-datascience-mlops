# Wrong number
curl -u alice:wonderland "http://127.0.0.1:8000/mcq?use=positioning%20test&subjects=databases,docker&num=11"

## should give:
{"detail":"Internal Server Error: 400: Invalid entry for 'num'. Choose from [5, 10, 20]"}

# Wrong use
curl -u alice:wonderland "http://127.0.0.1:8000/mcq?use=positioning&subjects=databases,docker&num=10" 

## should give:
{"detail":"Internal Server Error: 400: Invalid entry for 'use'. Allowed values: ['positioning test', 'validation test', 'total boot camp']"}

# Wrong subject
curl -u alice:wonderland "http://127.0.0.1:8000/mcq?use=positioning%20test&subjects=databases,dockers&num=11"

## should give: 
{"detail":"Internal Server Error: 400: Invalid subject(s): ['dockers']. Choose from ['databases', 'distributed systems', 'data streaming', 'docker']"}%  

# Not enough questions
curl -u alice:wonderland "http://127.0.0.1:8000/mcq?use=positioning%20test&subjects=databases,docker&num=20"

## should give:
{"detail":"Internal Server Error: 400: Not enough questions available. Only 11 questions available."}% 

# Correct curl
curl -u alice:wonderland "http://127.0.0.1:8000/mcq?use=positioning%20test&subjects=databases,docker&num=5"

## should give:
{"user":"alice","questions":[{"question":"Cassandra and HBase are databases","subject":"databases","use":"positioning test","correct":"C","responseA":"relational database","responseB":"object-oriented","responseC":"column-oriented","responseD":"graph-oriented"},{"question":"MongoDB and CouchDB are databases","subject":"databases","use":"positioning test","correct":"B","responseA":"relational database","responseB":"object-oriented","responseC":"column-oriented","responseD":"graph-oriented"},{"question":"Docker containers can communicate with each other using","subject":"docker","use":"positioning test","correct":"B","responseA":"volumes","responseB":"networks","responseC":"communications","responseD":""},{"question":"DockerHub is","subject":"docker","use":"positioning test","correct":"C","responseA":"a system that allows you to launch several containers at once","responseB":"a container orchestration system","responseC":"a Docker image directory","responseD":""},{"question":"Docker allows persisting changes","subject":"docker","use":"positioning test","correct":"C","responseA":"Yes","responseB":"No","responseC":"Yes provided you use volumes","responseD":""}]}%

# adding question as admin
curl -u admin:4dm1N -X POST "http://127.0.0.1:8000/admin" \
     -H "Content-Type: application/json" \
     -d '{
          "question": "What is 2 + 2?",
          "subject": "Math",
          "use": "Practice",
          "correct": "A",
          "responseA": "4",
          "responseB": "3",
          "responseC": "5",
          "responseD": "6",
          "remark": "Basic addition"
     }'

# should give
{"message":"Question added successfully and DataFrame saved!","new_row":{"question":"What is 2 + 2?","subject":"Math","use":"Practice","correct":"A","responseA":"4","responseB":"3","responseC":"5","responseD":"6","num":"NA","remark":"Basic addition"}}%  

# adding question wrong password
curl -u admin:4dm1N -X POST "http://127.0.0.1:8000/admin" \
     -H "Content-Type: application/json" \
     -d '{
          "question": "What is 2 + 2?",
          "subject": "Math",
          "use": "Practice",
          "correct": "A",
          "responseA": "4",
          "responseB": "3",
          "responseC": "5",
          "responseD": "6",
          "remark": "Basic addition"
     }'

# should give
{"detail":"Invalid credentials"}% 

# adding question, only two choices given
curl -u admin:4dm1n -X POST "http://127.0.0.1:8000/admin" \ 
     -H "Content-Type: application/json" \
     -d '{
          "question": "What is 2 + 2?",
          "subject": "Math",
          "use": "Practice",
          "correct": "A",
          "responseA": "4",
          "responseB": "5",
          "remark": "Basic addition"
     }'
     
# should give
{"message":"Question added successfully and DataFrame saved!","new_row":{"question":"What is 2 + 2?","subject":"Math","use":"Practice","correct":"A","responseA":"4","responseB":"5","responseC":"NA","responseD":"NA","remark":"Basic addition"}}%     
