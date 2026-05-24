from nltk.corpus import stopwords as stopwords_nltk
import stopwordsiso

from topic_modeling.utils.common_utils import read_stopwords


def get_stopwords(
        path_to_additional_stop_words : str
) -> list[str]:
    """
    Gets stop words, including the additional ones.

    Parameters:
    - path_to_additional_stop_words: path to the additional stop words list

    Returns:
    - all_stop_words: sorted list of the all unique stop words extracted from
    the two packages and the additional stop words list
    """
    slovenian_stopwords_nltk = stopwords_nltk.words('slovene')
    slovenian_stopwords_iso = list(stopwordsiso.stopwords("sl"))
    archaic_stopwords = read_stopwords(path_to_additional_stop_words)

    all_stopwords = sorted(list(set(
        slovenian_stopwords_iso + 
        slovenian_stopwords_nltk + 
        archaic_stopwords)))
    return all_stopwords
    

if __name__=="__main__":
    from topic_modeling.utils.common_utils import read_config

    config = read_config("topic_modeling/config/config.yaml")
    path_to_additional_stop_words = config["input_data"]["additional_stop_words_path"]

    all_stopwords = get_stopwords(path_to_additional_stop_words)
    print(f"The first 10 stop words: {all_stopwords[:10]}")