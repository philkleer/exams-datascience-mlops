import requests
import pytest

# urls and endpoints
login_url = "http://127.0.0.1:3000/login"
predict_url = "http://127.0.0.1:3000/v1/models/admission_regress/predict"


# JWT Tests
# missing JWT token
def test_missing_token():
    response = requests.post(predict_url, headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert "Missing authentication token" in response.text


# invalid JWT token
def test_invalid_token():
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer invalid_token",
    }
    response = requests.post(predict_url, headers=headers)
    assert response.status_code == 401
    assert "Invalid token" in response.text


# valid JWT token
@pytest.fixture
def valid_token():
    credentials = {"username": "user123", "password": "password123"}
    login_response = requests.post(
        login_url, headers={"Content-Type": "application/json"}, json=credentials
    )
    return login_response.json().get("token")


def test_valid_token(valid_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {valid_token}",
    }
    response = requests.post(
        predict_url,
        headers=headers,
        json={
            "gre_score": 320,
            "toefl_score": 120,
            "university_rating": 2,
            "cgpa": 3,
            "research": 1,
        },
    )
    assert response.status_code == 200
    assert "prediction" in response.json()


# Login API Tests
# valid credentials
def test_valid_credentials():
    credentials = {"username": "user123", "password": "password123"}
    response = requests.post(
        login_url, headers={"Content-Type": "application/json"}, json=credentials
    )
    assert response.status_code == 200
    assert "token" in response.json()


# invalid credentials
def test_invalid_credentials():
    credentials = {"username": "user123", "password": "wrongpassword"}
    response = requests.post(
        login_url, headers={"Content-Type": "application/json"}, json=credentials
    )
    # assert response.status_code == 401
    assert "401" in response.text


# Prediction Tests
# missing JWT token
def test_missing_token_prediction():
    response = requests.post(predict_url, headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert "Missing authentication token" in response.text


# invalid JWT token
def test_invalid_token_prediction():
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer invalid_token",
    }
    response = requests.post(
        predict_url,
        headers=headers,
        json={
            "gre_score": 320,
            "toefl_score": 120,
            "university_rating": 2,
            "cgpa": 3,
            "research": 1,
        },
    )
    assert response.status_code == 401
    assert "Invalid token" in response.text


# valid prediction
def test_valid_prediction(valid_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {valid_token}",
    }
    data = {
        "gre_score": 320,
        "toefl_score": 120,
        "university_rating": 2,
        "cgpa": 3,
        "research": 1,
    }
    response = requests.post(predict_url, headers=headers, json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()


# invalid input data
def test_invalid_input_data(valid_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {valid_token}",
    }
    invalid_data = {
        "gre_score": "invalid",
        "toefl_score": 120,
        "university_rating": 2,
        "cgpa": 3,
        "research": 1,
    }
    response = requests.post(predict_url, headers=headers, json=invalid_data)
    assert response.status_code == 400

    error_message = response.json()
    assert "Input should be a valid integer" in error_message
