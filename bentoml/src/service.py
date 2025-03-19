import numpy as np
import bentoml
import pickle
from bentoml.io import JSON
from pydantic import BaseModel, ValidationError
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {"user123": "password123", "user456": "password456"}


# defining authorization
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/admission_regress/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(
                    status_code=401, content={"detail": "Missing authentication token"}
                )

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(
                    status_code=401, content={"detail": "Token has expired"}
                )
            except jwt.InvalidTokenError:
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid token"}
                )

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response


# Pydantic model to validate input data
class InputModel(BaseModel):
    gre_score: int
    toefl_score: int
    university_rating: int
    cgpa: int
    research: int


# loading model
model_ref = bentoml.sklearn.get("linreg_admission:3ppk7ohgdgzubtqd")
# building runner
model_runner = model_ref.to_runner()
# loading scaler from model
scaler = pickle.loads(model_ref.custom_objects["scaler"])

# launching service
your_future = bentoml.Service("admission_service", runners=[model_runner])
your_future.add_asgi_middleware(JWTAuthMiddleware)


@your_future.api(input=JSON(), output=JSON(), route="/login")
def login(credentials: dict) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    elif username not in USERS:
        # this does not work: gives code 500
        return JSONResponse(
            status_code=401,
            content={"status": "error", "detail": "Invalid credentials"},
        )
    elif USERS[username] != password:
        # return JSONResponse(
        #     status_code=401,
        #     content={
        #         'status': 'error',
        #         'detail': 'Invalid credentials'
        #     }
        # )
        # tried plain dictionary
        return {"status": "error", "detail": "Invalid credentials"}, 401


@your_future.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route="/v1/models/admission_regress/predict",
)
async def classify(input_data: InputModel, ctx: bentoml.Context) -> dict:
    request = ctx.request
    user = request.state.user if hasattr(request.state, "user") else None

    try:
        input_series = np.array(
            [
                input_data.gre_score,
                input_data.toefl_score,
                input_data.university_rating,
                input_data.cgpa,
                input_data.research,
            ]
        ).reshape(1, -1)

        input_scaled = scaler.transform(input_series)

        result = await model_runner.predict.async_run(input_scaled)

        return {"prediction": result.tolist(), "user": user}

    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": f"Invalid input: {e.errors()}"},
        )


# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token
