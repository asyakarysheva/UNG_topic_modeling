import os
import spacy
from tqdm import tqdm
from pathlib import Path

from topic_modeling.utils.stop_words_extraction import get_stopwords
from topic_modeling.utils.common_utils import save_lemmas, read_lemmas


class Lemmatizer:
    """
    Lemmatizes the corpus
    """
    def __init__(
            self,
            config : dict,
            texts_to_lemmatize: list[dict],
            texts_with_other_encodings: list[dict]
    ):
        self.path_to_additional_stopwords = config["input_data"]["additional_stop_words_path"]
        self.output_dir = config["output_data"]["output_dir"]
        self.lemmatized_corpus_dir = Path(self.output_dir).joinpath(
            config["output_data"]["lemmatized_corpus_dir"]
        )
        os.makedirs(self.lemmatized_corpus_dir, exist_ok=True)

        #all_file_names = sorted(set(os.listdir(self.lemmatized_corpus_dir)))
        #self.all_text_paths = [
        #    f"{self.lemmatized_corpus_dir}/{file_name}" for file_name in all_file_names
        #]
        self.texts_to_lemmatize = texts_to_lemmatize
        self.texts_with_other_encodings = {
            text["file_name"] for text in texts_with_other_encodings
        }

        self.nlp = spacy.load("sl_core_news_sm")
        self.nlp.max_length = 5000000

        self.stopwords = get_stopwords(self.path_to_additional_stopwords)

    def lemmatize_text(
            self,
            text: str
    ) -> list[str]:
        """
        Lemmatizes slovenian text removing stop words.

        Parameters:
        - text: text to lemmatize

        Returns:
        - lemmas: the list of lemmas from the text
        """
        doc = self.nlp(text)
        lemmas = []

        for token in doc:
            if (token.pos_ == "NOUN" and
                not token.lemma_.lower() in self.stopwords and
                not token.is_stop and
                not token.ent_type_ and 
                "\n" not in token.lemma_):
                lemmas.append(token.lemma_.lower()) # keep only nouns
        # usage of another model
        # add metadata (date) to the topics
        # probability of the topic appearing in the text; presenting in graphs

        return lemmas
    
    def lemmatize_all_texts(
            self
    ) -> list[dict]:
        """
        Lemmatizes and saves all texts.
        Iterates over all the text names in the directory and either reads the texts
        that are already lemmatized or lemmatizes the ones that are not.

        Returns:
        - all_lemmatized_texts: the list of all lemmatized texts
        """
        all_lemmatized_texts = []

        # If all texts have already been lemmatized they can be simply read
        if not self.texts_to_lemmatize:
            for file_path in sorted(self.lemmatized_corpus_dir.iterdir()):
                if file_path.is_file():
                    lemmas = read_lemmas(file_path)
                    all_lemmatized_texts.append({
                        "file_name": file_path.name.removesuffix(".txt"),
                        "lemmas": lemmas
                    })
            return all_lemmatized_texts 

        # Otherwise texts will be lemmatized
        for text in tqdm(self.texts_to_lemmatize,
                         desc="Lemmatizing texts...",
                         total=len(self.texts_to_lemmatize)):
            
            file_name = text["file_name"]
            file_path = Path(self.lemmatized_corpus_dir).joinpath(f"{file_name}.txt")

            if file_name in self.texts_with_other_encodings:
                continue
            if file_path.exists():
                lemmas = read_lemmas(file_path)
            else:
                lemmas = self.lemmatize_text(text["text"])
                save_lemmas(lemmas, file_path)

            dict_ = {
                "file_name": file_name,
                "lemmas": lemmas
            }
            all_lemmatized_texts.append(dict_)
        return all_lemmatized_texts

    def __call__(self) -> list[dict]:
        """
        Lemmatizes the corpus

        Returns:
        - all_lemmatized_texts: the list of all lemmatized texts
        """
        return self.lemmatize_all_texts()
    

if __name__=="__main__":
    from topic_modeling.utils.common_utils import read_config
    from topic_modeling.pipeline.corpus_reader import TextsReader

    config = read_config("topic_modeling/config/config.yaml")

    texts_reader = TextsReader(config)
    all_texts, texts_with_other_encodings = texts_reader()
    text_names_with_other_encodings = [
        text["file_name"]
        for text in texts_with_other_encodings
    ]
    print(f"The names of the texts with other encodings: {text_names_with_other_encodings}")

    lemmatizer = Lemmatizer(config,
                            all_texts,
                            texts_with_other_encodings)
    lemmatized_texts = lemmatizer()

    print(f"First lemmatized text: {lemmatized_texts[0]}")
    print(f"Its author and title: {lemmatized_texts[0]['file_name']}")
    
    print("All texts have been lemmatized!")