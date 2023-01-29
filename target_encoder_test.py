import pytest
import joblib
from sklearn.preprocessing import LabelEncoder

@pytest.fixture
def target_encoder():
    # Load the target encoder from the file
    return joblib.load('./models/target_encoder.joblib')

def test_target_encoder(target_encoder):
    # Assert that the target encoder is an instance of sklearn's LabelEncoder
    assert isinstance(target_encoder, LabelEncoder)

    # Assert that the target encoder has the correct number of classes
    assert len(target_encoder.classes_) == 16

    # # Assert that the target encoder correctly encodes a sample input
    # assert target_encoder.transform(['ENTP']) == [1]

    # # Assert that the target encoder correctly decodes a sample output
    # assert target_encoder.inverse_transform([1]) == ['ENTP']

    # Assert that the target encoder correctly encodes inputs    
    mbti_types = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP', 'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
    expected = [14, 12,  8, 10, 15, 13,  9, 11,  7,  5,  1,  3,  6,  4,  0,  2]
    result = target_encoder.transform(mbti_types)
    assert result.tolist() == expected, f"Expected {expected}, but got {result.tolist()}"

    # # Assert output has the correct shape
    # test_data = ['ENFP']
    # result = encoder.transform(test_data)
    # assert result is not None
    # assert len(result) == 1