import pytest

from sentiment_analysis.Deep_Learning.bert import read_data

def test_read_input_file():
    """
    Test to Check whether the code reads a file correctly from a specified path and outputs correct columns
    """
    input_train_data = read_data("../../data/curated/reviews/cleaned_reviews.csv")
    req_cols = ('labels', 'text')

    assert 'labels' in tuple(input_train_data.columns)
    assert 'text' in input_train_data.columns
