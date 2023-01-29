
import pytest

from functions import clean_text


def test_clean_text():
    # Test that the function correctly removes URLs and non-alphanumeric characters from text
    text = ['This is a test sentence with a URL: https://www.example.com', 'Another sentence with a URL: www.example.com']
    cleaned_text = clean_text(text)
    assert cleaned_text == ['this is a test sentence with a url   ', 'another sentence with a url   ']















