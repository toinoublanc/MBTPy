# Importing packages & libraries
#---------------------------------------------------------
from fastapi import FastAPI, Query, Header, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
from enum import Enum
from typing import Optional, Union
from csv import DictReader
import random
import datetime


# API generation
#---------------------------------------------------------
api = FastAPI(
    title="FastAPI_eval_TB",
    description="API created by Toinou BLANC for FastAPI evaluation purpose",
    version="1.0.0",
    openapi_tags=[
        {
        'name': 'default',
        'description': 'functions accessible by any user'
        },
        {
        'name': 'admin',
        'description': 'functions accessible by admin user'
        }
    ]
)


# Security
#---------------------------------------------------------
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Data loading
#---------------------------------------------------------
csv_filepath = 'questions.csv'
with open(csv_filepath, 'r') as csv_file:
    dict_reader = DictReader(csv_file)
    questions_db = list(dict_reader)

users_db = {
    "alice": {
        "username": "alice", 
        "hashed_password": pwd_context.hash('wonderland')
    },

    "bob": {
        "username": "bob", 
        "hashed_password": pwd_context.hash('builder')
    },

    "clementine": {
        "username": "clementine", 
        "hashed_password": pwd_context.hash('mandarine')
    },

    "admin": {
        "username": "admin", 
        "hashed_password": pwd_context.hash('4dm1N')
    },
}


# HTTP Basic Auth
#---------------------------------------------------------
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if not(users_db.get(username)) or not(pwd_context.verify(credentials.password, users_db[username]['hashed_password'])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Classes definition
#---------------------------------------------------------
class TestType(str, Enum):
    test_de_positionnement = "Test de positionnement"
    test_de_validation = "Test de validation"
    total_bootcamp = "Total Bootcamp"

class Item(BaseModel):
    question: str
    subject: str
    use: TestType
    correct: str
    responseA: str
    responseB: str
    responseC: Optional[str] = None
    responseD: Optional[str] = None
    remark: Optional[str] = None

class MyException(Exception):
    def __init__(self,
                 name : str,
                 date: str):
        self.name = name
        self.date = date


# Handling errors & custom exceptions
#---------------------------------------------------------
responses = {
    400: {"description": "Bad Request. Check spelling and accepted values."},
    403: {"description": "Not enough privileges."},
}

@api.exception_handler(MyException)
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


# Routers
#---------------------------------------------------------
@api.get('/', tags=['default', 'admin'], name='Get home')
def get_home(
    username: str = Depends(get_current_user)
    ):
    '''Says Hello to authorized users only'''
    return "Welcome {}!".format(username)

#---------------------------------------------------------
@api.get('/questions', tags=['default', 'admin'], name='Get all questions')
def get_questions(
    username: str = Depends(get_current_user)
    ):
    '''Returns all questions from the database'''
    return questions_db

#---------------------------------------------------------
@api.get('/questions/{use}', tags=['default', 'admin'], name='Get questions matching specified parameters')
def get_use(
    use:TestType, 
    subjects: Union[str, None] = Query(default=None), 
    nb_questions:Optional[int]=5,
    username: str = Depends(get_current_user)
    ):
    '''Returns a set of questions from the database.
        Arguments :
            use (str): what kind of test is it required for (ex : "Test de positionnement")
            subjects (optional, str, default=None): if given, only specified subjects will be included
            nb_questions (optional, int, default=5) : if given, number of questions to return. 
            Please note that the maximum number of questions depends on the choosen query parameters.
    '''
    # Filtering by 'use
    questions = list(filter(lambda x: x.get('use') == use, questions_db))
    # Optional filtering by 'subjects'
    if subjects :
        questions = list(filter(lambda x: x.get('subject') in subjects, questions))
    # Selecting 'nb_questions'
    if nb_questions <= len(questions):
        random.shuffle(questions)
        selected_questions = questions[0:nb_questions]
        return {'route' : 'dynamic','data': selected_questions}
    else:
        raise HTTPException(
                status_code=404,
                detail= "Not enough questions matching this request."
            )

#---------------------------------------------------------
@api.put('/question', tags=['admin'], responses=responses, name='Add a new question to the database')
def put_question(
    question:Item,
    username: str = Depends(get_current_user)
    ):
    '''Add a new question to the database and return its details
    For admin user only
    '''
    if username == 'admin':
        new_question = {
            'question': question.question,
            'subject': question.subject,
            'use': question.use,
            'correct': question.correct,
            'responseA': question.responseA,
            'responseB': question.responseB,
            'responseC': question.responseC,
            'responseD': question.responseD,
            'remark': question.remark
        }
        questions_db.append(new_question)
        return new_question
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin rights required",
            headers={"WWW-Authenticate": "Basic"}
        )

#---------------------------------------------------------
@api.get('/coffee', tags=['default', 'admin'], name='Need some coffee?')
def get_custom_exception(
    username: str = Depends(get_current_user)
    ):
    '''Purposely triggers error 418 to check if API is functional
    '''
    raise MyException(
      name="coffee error",
      date=str(datetime.datetime.now())
    )


























