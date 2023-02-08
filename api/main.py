##############################################################################
# Import packages & libraries
from fastapi import FastAPI, Query, Header, Request, HTTPException, Security, Depends, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from typing import List, Optional, Union
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
from csv import DictReader
from enum import Enum
import datetime
import joblib
import os


# Custom functions / modules / packages
from processing import clean_text

#------------------------------------------------------------------------------
# Global variables
DATABASE_FILE = "database.db"
DATABASE_PATH = "./database/"
MODELS_PATH = "./saved_models/"

#------------------------------------------------------------------------------
# Load saved models 
preprocessing_pipeline = joblib.load(f"{MODELS_PATH}preprocessing_pipeline.joblib")
target_encoder = joblib.load(f"{MODELS_PATH}target_encoder.joblib")
model_svc = joblib.load(f"{MODELS_PATH}model_svc.joblib")
# model_cat = joblib.load(f"{MODELS_PATH}model_cat.joblib")
# model_xgb = joblib.load(f"{MODELS_PATH}model_xgb.joblib")

##############################################################################
# Classes definition

class MBTIType(str, Enum):
    ISTJ = "ISTJ"
    ISFJ = "ISFJ"
    INFJ = "INFJ"
    INTJ = "INTJ"
    ISTP = "ISTP"
    ISFP = "ISFP"
    INFP = "INFP"
    INTP = "INTP"
    ESTP = "ESTP"
    ESFP = "ESFP"
    ENFP = "ENFP"
    ENTP = "ENTP"
    ESTJ = "ESTJ"
    ESFJ = "ESFJ"
    ENFJ = "ENFJ"
    ENTJ = "ENTJ"

class Role(str, Enum):
    admin = "admin"
    standard = "standard"

# class ModelType(str, Enum):
#     model_svc = "model_svc"
#     model_cat = "model_cat"
#     model_xgb = "model_xgb"

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    password_hash: str
    role: Role
    first_name: Optional[str]
    last_name: Optional[str]
    predictions: List["Prediction"] = Relationship(back_populates="users")

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    input_text: str
    predicted_type: str
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    users: Optional[User] = Relationship(back_populates="predictions")

class MyException(Exception):
    def __init__(self,
                 name : str,
                 date: str):
        self.name = name
        self.date = date


##############################################################################
# Database setup

sqlite_file_path = DATABASE_PATH + DATABASE_FILE
sqlite_url = f"sqlite:///{sqlite_file_path}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=False, connect_args=connect_args)
# remove 'echo' in production

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# Dependency functions
def get_session():
    with Session(engine) as session:
        yield session


##############################################################################
# Security

security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    with Session(engine) as session:
        username = credentials.username
        user = session.query(User).filter_by(username=username).first()
        if not user or not pwd_context.verify(credentials.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
    return username



def get_role_admin(*, 
    session: Session = Depends(get_session),
    credentials: HTTPBasicCredentials = Depends(security), 
    role: Role,
    ):
    username = credentials.username
    user = session.query(User).filter_by(username=username).first()
    role = user.role

    if not role == "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin rights required",
            headers={"WWW-Authenticate": "Basic"},
        )
    return role




#------------------------------------------------------------------------------

# API_KEY = os.environ.get("API_KEY")
API_KEY = "apikey1234"
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate api key"
    )


##############################################################################
# FastAPI app setup

app = FastAPI(
    title="MBTPy",
    description="'MBTI' Personnality-Type Indicator prediction API",
    version="1.1.0",
    openapi_tags=[
        {'name': 'user', 'description': 'routers related to user management'},
        {'name': 'prediction', 'description': 'routers related to predictions'},
        {'name': 'test', 'description': 'routers related to API testing'},
    ]
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# Handling errors & custom exceptions
responses = {
    400: {"description": "Bad Request. Check spelling and accepted values."},
    403: {"description": "Not enough privileges."},
}

@app.exception_handler(MyException)
def MyExceptionHandler(
    request: Request,
    exception: MyException
    ):
    '''Defines what to do when raising MyException
    '''
    return JSONResponse(
        status_code=418,
        content={
            'url': str(request.url),
            'name': exception.name,
            'message': "I'm a teapot",
            'date': exception.date
        }
    )



##############################################################################
# Routers


#------------------------------------------------------------------------------
# User management

@app.get("/user/all", tags=['user'])
def read_users(*, 
    session: Session = Depends(get_session),
    role: Role = Depends(get_role_admin),
    offset: int = 0, 
    limit: int = Query(default=100, lte=100),
    ):
    '''Read all users from the table `user` of the database.
    '''
    users = session.exec(select(User).offset(offset).limit(limit)).all()
    return users

@app.post("/user/new", tags=['user'])
def create_user(*, 
    session: Session = Depends(get_session),
    role: Role = Depends(get_role_admin), 
    user: User,
    ):
    '''Create a new user in the table `user` of the database.
    '''
    new_user = User.from_orm(user)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user

@app.get("/user/{user_id}", tags=['user'])
def read_user(*, 
    session: Session = Depends(get_session),
    role: Role = Depends(get_role_admin), 
    user_id: int,
    ):
    '''Read a specific user from the table `user` of the database.
    '''
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.delete("/user/{user_id}", tags=['user'])
def delete_user(*, 
    session: Session = Depends(get_session), 
    role: Role = Depends(get_role_admin),
    user_id: int,
    ):
    '''Delete a specific user from the table `user` of the database.
    '''
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    session.delete(user)
    session.commit()
    return {"ok": True}

@app.patch("/user/{user_id}", tags=['user'])
def update_user(*, 
    session: Session = Depends(get_session), 
    role: Role = Depends(get_role_admin),
    user_id: int, 
    user: User,
    ):
    '''Update a specific user from the table `user` of the database.
    '''
    db_user = session.get(User, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    user_data = user.dict(exclude_unset=True)
    for key, value in user_data.items():
        setattr(db_user, key, value)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


#------------------------------------------------------------------------------
# MBTI Prediction

@app.get("/prediction/all", tags=['prediction'])
def read_predictions(*, 
    session: Session = Depends(get_session), 
    username: str = Depends(get_current_user),
    offset: int = 0, 
    limit: int = Query(default=100, lte=100),
    ):
    '''Read all previous predictions stored in the table `prediction` of the database.
    '''
    predictions = session.exec(select(Prediction).offset(offset).limit(limit)).all()
    return predictions


@app.post("/prediction/type", tags=['prediction'])
def predict_type(*, 
    session: Session = Depends(get_session), 
    text: str, 
    api_key_header: APIKey = Depends(get_api_key)
    ):
    '''Predict the 4-letters personnality type from a sample of text and store it in the table `prediction` of the database.
    '''
    text_data = [text]
    cleaned_text = clean_text(text_data)
    preprocessed_text = preprocessing_pipeline.transform(cleaned_text)
    prediction_raw = model_svc.predict(preprocessed_text)
    predicted_type = target_encoder.inverse_transform(prediction_raw)[0]
    prediction = Prediction(
        input_text=text, 
        model='model_svc',
        predicted_type=predicted_type
        )
    session.add(prediction)
    session.commit()
    session.refresh(prediction)

    return {"predicted_type": predicted_type}



#------------------------------------------------------------------------------
# Test

@app.get("/home", tags=['test'])
def home(
    ):
    '''Basic endpoint.
    '''
    return {"message": "Hello ! "}


@app.get('/auth', tags=['test'])
def check_auth(
    username: str = Depends(get_current_user),
    ):
    '''Says Hello to authentified users only
    '''
    return "Welcome {}!".format(username)


@app.get('/auth/admin', tags=['test'])
def check_admin(
    username: str = Depends(get_current_user),
    role: Role = Depends(get_role_admin),
    ):
    '''Says Hello to admin users only
    '''
    return "Welcome {}!".format(username)


@app.get("/key", tags=['test'])
async def check_apikey(
    api_key_header: APIKey = Depends(get_api_key),
    ):
    '''Check API Key validity.
    '''
    return 'Hello this is secure ! '


@app.post("/user/demo", tags=['test'])
def create_demo_users(*, 
    session: Session = Depends(get_session)
    ):
    '''Fill the `user` table of the database with 4 demo users (1 'admin' and 3 'standard').
    '''
    demo_user_1 = User(
        username = "Neo", 
        password_hash = pwd_context.hash('admin'),
        role = "admin",
        first_name = "Thomas",
        last_name = "Anderson",
    )
    demo_user_2 = User(
        username = "Arnold", 
        password_hash = pwd_context.hash('hastalavista'),
        role = "standard",
        first_name = "T1000",
        last_name = "Governator",
    )
    demo_user_3 = User(
        username = "Sarah", 
        password_hash = pwd_context.hash('sarahpwd'),
        role = "standard",
        first_name = "Sarah",
        last_name = "Canard",
    )
    demo_user_4 = User(
        username = "John", 
        password_hash = pwd_context.hash('johnpwd'),
        role = "standard",
        first_name = "John",
        last_name = "Canard",
    )
    session.add(demo_user_1)
    session.add(demo_user_2)
    session.add(demo_user_3)
    session.add(demo_user_4)
    session.commit()
    session.refresh(demo_user_4)
    return {"ok": True}             


@app.get('/test/coffee', tags=['test'], name='Need some coffee?')
def get_custom_exception(*, 
    session: Session = Depends(get_session)
    ):
    '''Purposely triggers error 418 to check if API is functional.
    '''
    raise MyException(
      name="coffee error",
      date=str(datetime.datetime.now())
    )




