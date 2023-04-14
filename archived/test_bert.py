import pytest

from sentiment_analysis.Deep_Learning.bert import read_data, labels_array, tokenization

@pytest.fixture
def path_to_input_data():
    path = "../../data/curated/reviews/cleaned_reviews.csv"
    return path

def test_read_input_file(path_to_input_data):
    """
    Test to Check whether the code reads a file correctly from a specified path and outputs correct columns
    """
    input_train_data = read_data(path_to_input_data)

    assert 'labels' in input_train_data.columns
    assert 'text' in input_train_data.columns

def test_labels_tokenization(path_to_input_data):
    """Test to check output of labels_array and tokenization function output have the same length
    """
    data = read_data(path_to_input_data)
    labels = labels_array(data)
    Xids, Xmask = tokenization(data)
    assert len(labels) == len(Xids)
    assert len(labels) == len(Xmask)
    assert len(Xmask) == len(Xids)


