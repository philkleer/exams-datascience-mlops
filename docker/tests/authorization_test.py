import os
import requests


def test_access(username, password, access, expected_status):
    response = requests.get(
        url="http://api:8000{access}".format(access=access),
        params={"username": username, "password": password},
    )

    # get status
    status_code = response.status_code

    # display the results
    test_status = "SUCCESS" if str(status_code) == expected_status else "FAILURE"

    output = f"""


    ============================
        Authorization Test
    ============================
    User: {username}, Endpoint: {access}
    Expected: {expected_status}, Got: {status_code}
    ==> {test_status}

    ============================
           End of Test
    ============================

    
    """
    print(output)

    if os.getenv("LOG") == "1":
        with open("/logs/logs.txt", "a") as file:
            file.write(output + "\n")


# Tests
# Getting environment variables
USER1 = os.environ.get("USER1")
PASS1 = os.environ.get("PASS1")
USER2 = os.environ.get("USER2")
PASS2 = os.environ.get("PASS2")
AREA1 = os.environ.get("AREA1")
AREA2 = os.environ.get("AREA2")
RESULT1 = os.environ.get("RESULT1")
RESULT2 = os.environ.get("RESULT2")
RESULT3 = os.environ.get("RESULT3")
RESULT4 = os.environ.get("RESULT4")

# first test
test_access(USER1, PASS1, AREA1, RESULT1)

# second test
test_access(USER1, PASS1, AREA2, RESULT2)

# third test
test_access(USER2, PASS2, AREA1, RESULT3)

# fourth test
test_access(USER2, PASS2, AREA2, RESULT4)
