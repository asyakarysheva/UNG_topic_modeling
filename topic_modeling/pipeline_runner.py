from gensim.models.ldamodel import LdaModel

from topic_modeling.utils.common_utils import read_config
from topic_modeling.pipeline.corpus_reader import TextsReader
from topic_modeling.pipeline.lemmatizer import Lemmatizer
from topic_modeling.pipeline.lda_topic_modeling import LDATopicModeling


class LDATopicModelingPipeline:
    """
    Full pipeline for LDA topic modeling of the corpus
    """
    def __init__(self):
        self.config = read_config("topic_modeling/config/config.yaml")

    def read_texts(self) -> tuple[list[dict], list[dict]]:
        """
        Reads texts and identifies texts with the encoding different from utf-8 or iso-8859-2

        Returns:
        - all_texts: dictionary containing the texts from the corpus and their metadata
        - texts_with_other_encodings: texts with the encodings different from 'utf-8'
        """
        texts_reader = TextsReader(self.config)
        all_texts, texts_with_other_encodings = texts_reader()
        return all_texts, texts_with_other_encodings
    
    def lemmatize_texts(
            self,
            all_texts: list[dict],
            texts_with_other_encodings: list[dict]
    ) -> list[dict]:
        """
        Lemmatizes the corpus

        Returns:
        - all_lemmatized_texts: the list of all lemmatized texts
        """
        lemmatizer = Lemmatizer(self.config,
                                all_texts,
                                texts_with_other_encodings)
        lemmatized_texts = lemmatizer()
        return lemmatized_texts
    
    def perform_lda_topic_modeling(
            self,
            lemmatized_texts: list[dict]
    ) -> LdaModel:
        """
        Runs the topic modeling pipeline

        Returns:
        - best_model: trained LDA Model corresponding to the highest coherence value
        """
        lda_topic_modeling = LDATopicModeling(self.config, lemmatized_texts)
        best_model = lda_topic_modeling()
        return best_model
    
    def __call__(self):
        """
        Runs the full pipeline for LDA topic modeling of the corpus
        """
        all_texts, texts_with_other_encodings = self.read_texts()
        lemmatized_texts = self.lemmatize_texts(all_texts, texts_with_other_encodings)
        best_model = self.perform_lda_topic_modeling(lemmatized_texts)
        return best_model
    

if __name__=="__main__":
    lda_topic_modeling_pipeline = LDATopicModelingPipeline()
    best_model = lda_topic_modeling_pipeline()