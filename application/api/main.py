#------------------------------------------------------------------------------
# Importing packages & libraries

from fastapi import FastAPI
import joblib
import re
from tqdm import tqdm


#------------------------------------------------------------------------------
# API generation

app = FastAPI(
    title="MBTPy",
    description="API created by Toinou BLANC for MLOps Project",
    version="0.2.0",
)



#------------------------------------------------------------------------------

def clean_text(text):
    cleaned_text=[]
    for sentence in tqdm(text):
        sentence=sentence.lower()
        sentence=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',sentence)
        sentence=re.sub('[^0-9a-z]',' ',sentence)
        cleaned_text.append(sentence)
    return cleaned_text


preprocessing_pipeline = joblib.load('./models/preprocessing_pipeline.joblib')
target_encoder = joblib.load('./models/target_encoder.joblib')
model = joblib.load('./models/model_svc.joblib')



#------------------------------------------------------------------------------
# Routers

@app.get("/")
def home():
    return {"message": "Hello ! "}


@app.post("/predict")
def predict(text: str):
    text_data = [text]
    cleaned_text = clean_text(text_data)
    preprocessed_text = preprocessing_pipeline.transform(cleaned_text)
    prediction = model.predict(preprocessed_text)
    predicted_type = target_encoder.inverse_transform(prediction)[0]

    # session = Session()
    # prediction = Prediction(text=text, predicted_type=predicted_type)
    # session.add(prediction)
    # session.commit()
    # session.close()

    return {"predicted_type": predicted_type}





