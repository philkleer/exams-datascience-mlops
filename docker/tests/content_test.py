import os
import requests

# caplog is an inherent log from pytest
def test_sentence(sentence, access, expected_score):
    response = requests.get(
        url='http://api:8000{access}'.format(access=access),
        params= {
            'username': 'alice',
            'password': 'wonderland',
            'sentence': sentence
        }
    )
    score = response.json().get('score', None)
    status = ''
    if expected_score == '1':
        status = 'SUCCESS' if score > 0 else 'FAILURE'
    elif expected_score == '-1':
        status = 'SUCCESS' if score < 0 else 'FAILURE'
    
    output = f'''


    ============================
        Sentiment Analysis Test
    ============================
    User: alice, Sentence: wonderland
    Sentence: {sentence} 
    Expected: {expected_score}, Got: {score}
    ==> {status}

    ============================
           End of Test
    ============================

    
    '''
    print(output)

    if os.getenv('LOG') == '1':
        with open('/logs/logs.txt', 'a') as file:
            file.write(output + '\n')
            

# Getting environment variables
SENTENCE1 = os.environ.get('SENTENCE1')
SENTENCE2 = os.environ.get('SENTENCE2')
ACCESS1 = os.environ.get('ACCESS1')
ACCESS2 = os.environ.get('ACCESS2')
RESULT1 = os.environ.get('RESULT1')
RESULT2 = os.environ.get('RESULT2')

# First test
test_sentence(SENTENCE1, ACCESS1, RESULT1)

# Second test
test_sentence(SENTENCE1, ACCESS2, RESULT1)

# Third test
test_sentence(SENTENCE2, ACCESS1, RESULT2)

# Fourth test
test_sentence(SENTENCE2, ACCESS2, RESULT2)