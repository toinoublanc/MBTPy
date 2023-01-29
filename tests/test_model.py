import joblib
import pytest

def test_model_cat():
    model = joblib.load('./models/model_cat.joblib')
    test_data = ["'World is a beautiful place full of amazing people and opportunities ! ||| Hello there !"]

    result = model.predict(test_data)

    assert result is not None
    assert len(result) == 1
