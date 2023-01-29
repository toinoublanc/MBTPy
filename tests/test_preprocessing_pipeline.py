import joblib
import pytest
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError


# def test_clean_text():
#     # Test that the function correctly removes URLs and non-alphanumeric characters from text
#     text = ['This is a test sentence with a URL: https://www.example.com', 'Another sentence with a URL: www.example.com']
#     cleaned_text = clean_text(text)
#     assert cleaned_text == ['this is a test sentence with a url', 'another sentence with a url']

def test_preprocessing_pipeline():
    # Load the preprocessing pipeline from the file
    preprocessing_pipeline = joblib.load('./models/preprocessing_pipeline.joblib')

    # Test that the pipeline is not fitted before calling transform
    with pytest.raises(NotFittedError):
        preprocessing_pipeline.transform(['This is a test sentence'])
        
    # Fit the pipeline on some sample data
    train_data = ['This is a test sentence', 'Another test sentence']
    preprocessing_pipeline.fit(train_data)
    
    # Test that the pipeline correctly preprocesses the data
    test_data = ['This is another test sentence']
    transformed_data = preprocessing_pipeline.transform(test_data)
    # Assert that the transformed data has the correct shape
    assert transformed_data.shape == (1, 2)
