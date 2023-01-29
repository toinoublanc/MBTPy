import pytest
import joblib
from sklearn.preprocessing import LabelEncoder

def test_target_encoder():
    # Load the target encoder from the file
    target_encoder = joblib.load('./models/target_encoder.joblib')

    # Assert that the target encoder is an instance of sklearn's LabelEncoder
    assert isinstance(target_encoder, LabelEncoder)

    # Assert that the target encoder has the correct number of classes
    assert len(target_encoder.classes_) == 16

    # Assert that the target encoder correctly encodes a sample input
    assert target_encoder.transform(['ENTP']) == [1]

    # Assert that the target encoder correctly decodes a sample output
    assert target_encoder.inverse_transform([1]) == ['ENTP']


# import joblib
# import pytest

# def test_target_encoder():
#     encoder = joblib.load('./models/target_encoder.joblib')
#     test_data = ['ENFP']

#     result = encoder.transform(test_data)

#     assert result is not None
#     assert len(result) == 1

