from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
import pandas as pd

# Import your trained model
from model import predict_mbti_type

app = FastAPI()

# Connect to the database
engine = create_engine("YOUR DATABASE URL")

@app.post("/predict")
async def predict(text: str):
    try:
        # Preprocess the input text
        cleaned_text = clear_text(text)
        tokenized_text = tokenize_text(cleaned_text)
        vectorized_text = preprocessing_pipeline.transform(tokenized_text)

        # Use the model to make a prediction
        prediction = predict_mbti_type(vectorized_text)
        
        # Add prediction to the database
        df = pd.DataFrame({'text': text, 'prediction': prediction}, index=[0])
        df.to_sql('predictions', con=engine, if_exists='append', index=False)
        
        return JSONResponse(content={"status": "success", "prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# This is a simple example, but you will probably want to adjust it to suit your needs. This example uses the function clear_text, tokenize_text, preprocessing_pipeline for data preprocessing and predict_mbti_type for model prediction. Also, it saves the prediction and the text in the 'prediction' table in the sql database.

# You will need to replace "YOUR DATABASE URL" with the actual URL for your database and make sure that you have the necessary dependencies installed, such as SQLAlchemy for connecting to the database and FastAPI for creating the API.



# Once you have a trained model, you can then use the FastAPI library to create an API endpoint that takes user input, applies the preprocessing pipeline, and then makes a prediction using the trained model. You can also use SQLAlchemy to interact with a SQL database to store the user input and the corresponding predictions.
# Here is an example of how you could set up the API using FastAPI:

from fastapi import FastAPI, Request, Form
from sklearn.externals import joblib

app = FastAPI()

# Load the preprocessing pipeline and trained model
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
model = joblib.load("trained_model.pkl")

@app.post("/predict")
async def predict(request: Request, text: str = Form(...)):
    # Apply preprocessing pipeline to user input
    preprocessed_text = preprocessing_pipeline.transform([text])
    # Make prediction using the trained model
    prediction = model.predict(preprocessed_text)
    return {"prediction": prediction}
