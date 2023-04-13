import pytest

from sentiment_analysis.Deep_Learning.distilbert import scoring_single_review, load_distilbert_model

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
    



