import pytest
import joblib


def test_model_svc():
    # Loading the trained model and pipelines
    model = joblib.load('./models/model_svc.joblib')
    preprocessing_pipeline = joblib.load('./models/preprocessing_pipeline.joblib')
    target_encoder = joblib.load('./models/target_encoder.joblib')

    # new_data is the data we want to preprocess, it should be a list of text.
    new_data = ["'World is a beautiful place full of amazing people and opportunities ! ||| Hello there !"]

    # Clean and preprocess the text data
    new_post = preprocessing_pipeline.transform(new_data)

    # Using the model to predict the personality type
    result = model.predict(new_post)
    assert result is not None
    assert len(result) == 1

    mbti_types = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP', 'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
    predicted_type = target_encoder.inverse_transform(result)
    assert predicted_type[0] in mbti_types
