import os
import requests


def authentication_test(username, password, expected_status):
    # get response
    response = requests.get(
        url="http://api:8000/permissions",
        params={"username": username, "password": password},
    )
    # query status
    status_code = response.status_code

    # display the results
    test_status = "SUCCESS" if str(status_code) == expected_status else "FAILURE"

    output = f"""


    ============================
        Authentication test
    ============================
    request done at '/permissions'
    | username={username}
    | password={password}
    expected result = {expected_status}
    actual restult = {status_code}
    ==>  {test_status}

    ============================
           End of Test
    ============================


    """
    print(output)

    # printing in a file
    if os.getenv("LOG") == "1":
        with open("/logs/logs.txt", "a") as file:
            file.write(output + "\n")


# Creating tests
# Getting environment variables
USER1 = os.environ.get("USER1")
PASS1 = os.environ.get("PASS1")
RESULT1 = os.environ.get("RESULT1")
USER2 = os.environ.get("USER2")
PASS2 = os.environ.get("PASS2")
RESULT2 = os.environ.get("RESULT2")
USER3 = os.environ.get("USER3")
PASS3 = os.environ.get("PASS3")
RESULT3 = os.environ.get("RESULT3")

# First test
authentication_test(USER1, PASS1, RESULT1)

# Second test
authentication_test(USER2, PASS2, RESULT2)

# Third test
authentication_test(USER3, PASS3, RESULT3)
