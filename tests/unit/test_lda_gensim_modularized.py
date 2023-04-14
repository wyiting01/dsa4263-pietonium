import pytest
import random

from topic_modelling.combine_modularized import *

@pytest.fixture
def load_data():
    df_path = '../../data/curated/reviews/cleaned_reviews.csv'
    clean_data = read_data(df_path)
    return clean_data


def test_combine_reviews_to_list(load_data):
    """
    Function to test the output of the function is a list and
    consists of all rows in the data
    """
    data = combine_reviews_to_list(load_data)

    assert isinstance(data, list)
    assert len(data) == len(load_data)


def test_remove_top_10_words(load_data):
    """
    Function to check that top 10 words are removed from the texts
    """
    data = combine_reviews_to_list(load_data)
    data_words_before_removal = split_sentence(data)
    
    remove_lst = get_top_10_words(data_words_before_removal)
    data_words_after_removal = remove_top_10_words(data_words_before_removal,remove_lst)
    
    # random sampling to get indexes and list of words for comparision
    index_value = random.sample(list(enumerate(data_words_before_removal)), 2)
    first_index, second_index = index_value[0][0],index_value[1][0]
    
    assert len(data_words_before_removal[first_index]) >= len(data_words_after_removal[first_index])
    assert len(data_words_before_removal[second_index]) >= len(data_words_after_removal[second_index])


