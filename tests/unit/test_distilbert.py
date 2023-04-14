import pytest

from sentiment_analysis.Deep_Learning.distilbert import scoring_single_review, load_distilbert_model, scoring_file_dummy

@pytest.fixture
def load_model():
    model, tokenizer = load_distilbert_model()
    return model, tokenizer


def test_scoring_single_review(load_model):
    """
    Function to test the correctness of the scoring function
    """
    model, tokenizer = load_model
    raw_review1 = "I highly recommend this product. I am very pleased with it."
    raw_review2 = "I am disappointed in this product and it did not work as promised."

    raw_review1_pred = scoring_single_review(raw_review1, model, tokenizer)
    raw_review2_pred = scoring_single_review(raw_review2, model, tokenizer)

    assert raw_review1_pred[0] == 1
    assert raw_review2_pred[0] == 0


def test_scoring_file_dummy(load_model):
    """
    Function to test the output df of the function
    """
    model, tokenizer = load_model
    path = "../../data/curated/reviews/cleaned_reviews_copy.csv"
    returned_df = scoring_file_dummy(path, model, tokenizer)
    assert isinstance(returned_df.predicted_sentiment[0], str)
    assert isinstance(returned_df.predicted_sentiment_probability[0], float)
    



