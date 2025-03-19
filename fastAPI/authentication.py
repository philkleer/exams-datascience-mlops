# authentication.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

# Hardcoded user credentials
USER_CREDENTIALS = {"alice": "wonderland", "bob": "builder", "clementine": "mandarine"}

ADMIN_CREDENTIALS = {"admin": "4dm1n"}


def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password

    if username not in USER_CREDENTIALS or USER_CREDENTIALS[username] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password

    if username not in ADMIN_CREDENTIALS or ADMIN_CREDENTIALS[username] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username
