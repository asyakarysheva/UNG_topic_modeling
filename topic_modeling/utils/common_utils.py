import yaml
from pathlib import Path
import pickle


def read_config(
        path_to_config : str
) -> dict:
    """
    Reads yaml config file.

    Parameters:
    - path_to_config : path to configuration directory

    Returns:
    - config : configuration file as a dictionary
    """
    with open(path_to_config) as file:
        config = yaml.safe_load(file)
    return config

def read_stopwords(
        path_to_stop_words: str
) -> list[str]:
    """
    Reads archaic (additional) stop words.

    Parameters:
    - path_to_stop_words: path to the additional stop words list

    Returns:
    - additional_stop_words: the list of archaic (or other additional) stop words
    """
    with open(path_to_stop_words) as file:
        additional_stop_words = file.read().splitlines()
    return additional_stop_words

def save_lemmas(
        lemmas: list[str],
        file_path : Path
) -> None:
    """
    Saves a lemmatized text.

    Parameters:
    - lemmas: the list of lemmas
    - file_path: path to the file where lemmas will be stored
    """
    all_lemmas = " ".join(lemmas)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(all_lemmas)

def read_lemmas(
        file_path : Path
) -> list[str]:
    """
    Reads lemmas from a text file.

    Parameters:
    - file_path: path to a file where the lemmas are stored

    Returns:
    - lemmas_text: lthe list with lemmas
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lemmas_text = file.read().split()
    return lemmas_text

def save_corpus(
        corpus,
        output_path
) -> None:
    """
    Saves Gensim corpus

    Parameters:
    - corpus: Gensim corpus
    - output_path: path to write Gensim corpus
    """
    with open(output_path, "wb") as file:
        pickle.dump(corpus, file)