from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from topic_modeling.utils.common_utils import save_corpus


class LDATopicModeling:
    """
    Class for the topic modeling pipeline:
    - trains topic models and computes coherence values for various number of topics;
    - visualizes and saves the plot with the coherence values;
    - saves document topic distribution for every trained model;
    - identifies the best model, according to coherence values;
    - saves document topic distribution for the best model.
    """
    def __init__(
            self,
            config : dict,
            lemmatized_texts : list[dict]
    ):
        self.lda_config = config["lda_parameters"]
        self.limit = self.lda_config["limit"]
        self.start = self.lda_config["start"]
        self.step = self.lda_config["step"]
        self.num_of_topic_words = self.lda_config["num_of_topic_words"]
        self.num_of_topics_per_document = self.lda_config["num_of_topics_per_document"]

        self.metadata_path = config["input_data"]["metadata"]
        self.metadata = pd.read_excel(self.metadata_path)

        self.output_dir = config["output_data"]["output_dir"]
        self.coherence_figure_path = Path(self.output_dir).joinpath(
            config["output_data"]["coherence_figure_file_name"]
        )
        self.lda_model_path = Path(self.output_dir).joinpath(
            config["output_data"]["lda_model_name"]
        )
        self.id2word_path = Path(self.output_dir).joinpath(
            config["output_data"]["id2word_name"]
        )
        self.corpus_path = Path(self.output_dir).joinpath(
            config["output_data"]["corpus"]
        )

        self.text_names = [lemmatized_text["file_name"] 
                      for lemmatized_text in lemmatized_texts]
        self.texts = [lemmatized_text["lemmas"] 
                      for lemmatized_text in lemmatized_texts]
        self.id2word = Dictionary(self.texts)
        self.id2word.filter_extremes(no_below=5, no_above=0.8)
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]

        self.id2word.save(str(self.id2word_path))
        save_corpus(self.corpus, self.corpus_path)

    def compute_coherence_values(self) -> tuple[list]:
        """
        Computes coherence values for various number of topics.

        Returns:
        - model_list : list of LDA topic models
        - coherence_values : coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []

        for num_topics in tqdm(range(self.start, self.limit, self.step),
                               desc="Training the topic model and computing coherence values...",
                               total=len(range(self.start, self.limit, self.step))):
            model = LdaModel(
                corpus=self.corpus,
                id2word=self.id2word,
                num_topics=num_topics,
                random_state=100,
                update_every=1,
                passes = 5,
                chunksize=500,
                alpha='auto',
                per_word_topics=True
            )
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model, 
                texts=self.texts, 
                dictionary=self.id2word, 
                coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values
    
    def visualize_coherence(
            self,
            coherence_values : list[float]
    ) -> None:
        """
        Plots coherence values and saves the figure.

        Parameters:
        - coherence_values: list of coherence values
        """
        x = range(self.start, self.limit, self.step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.title("Coherence values")

        plt.savefig(self.coherence_figure_path, dpi=300, bbox_inches="tight")

    def save_single_model_topics_to_json(
            self,
            model: LdaModel
    ) -> None:
        """
        Saves topics of a single LDA model to Excel.
        One Excel file per model.

        Parameters:
        - model: trained LDA Model
        """
        num_topics = model.num_topics

        keys_list = ["topic_number"] + [f"word_{i+1}" for i in range(self.num_of_topic_words)]
        result_dict = {key: [] for key in keys_list}

        for topic_id in range(num_topics):
            result_dict["topic_number"].append(topic_id+1)
            topic_words = model.show_topic(topic_id, topn=self.num_of_topic_words)
            for i, word_weight in enumerate(topic_words):
                word, weight = word_weight
                rounded_weight = round(float(weight), 4)
                result_dict[f"word_{i+1}"].append((word, rounded_weight))
       
        result_df = pd.DataFrame(result_dict)

        output_path = Path(self.output_dir).joinpath(f"word_topic_distribution_{num_topics}.xlsx")
        result_df.to_excel(output_path)

    def save_document_topic_distribution(
            self,
            model: LdaModel,
            suffix: str = ""
    ) -> None:
        """
        Saves document-topic distribution to Excel.
        One Excel file per model.

        Parameters:
        - model: trained LDA Model
        """
        data = []

        for doc_name, bow in zip(self.text_names, self.corpus):
            topic_distribution = model.get_document_topics(bow, minimum_probability=0)
            topic_distribution = sorted(
                topic_distribution,
                key=lambda x: x[1],
                reverse=True
            )[:self.num_of_topics_per_document]
            topic_distribution = [
                (topic_id, round(float(prob), 4))
                for topic_id, prob in topic_distribution
            ]

            if doc_name in self.metadata["author_and_title"].values:
                current_row = self.metadata.loc[
                    self.metadata["author_and_title"] == doc_name
                ]
                shame_num = int(current_row["shame_num"].item())
                publication_year = int(current_row["publ_year_clean"].item())
            else:
                shame_num = "not_in_corpus"
                publication_year = "not_in_corpus"

            data.append({
                "shame_num": shame_num,
                "text_file_name": doc_name,
                "publication_year": publication_year,
                "topics": topic_distribution
            })

        df = pd.DataFrame(data)

        output_path = Path(self.output_dir).joinpath(
            f"doc_topic_distribution_{model.num_topics}{suffix}.xlsx")
        df.to_excel(output_path, index=False)

    def __call__(self) -> LdaModel:
        """
        Runs the topic modeling pipeline:
        - trains topic models and computes coherence values for various number of topics;
        - visualizes and saves the plot with the coherence values;
        - saves document topic distribution for every trained model;
        - identifies the best model, according to coherence values;
        - saves document topic distribution for the best model.

        Returns:
        - best_model: trained LDA Model corresponding to the highest coherence value
        """
        model_list, coherence_values = self.compute_coherence_values()
        
        self.visualize_coherence(coherence_values)
        for model in model_list:
            self.save_single_model_topics_to_json(model)
            self.save_document_topic_distribution(model)

        # Identifying the best model
        best_index = coherence_values.index(max(coherence_values))
        best_model = model_list[best_index]
        best_num_topics = best_model.num_topics
        best_coherence = coherence_values[best_index]

        print(f"Best model: {best_num_topics} topics")
        print(f"Best coherence: {round(best_coherence, 4)}")

        self.save_document_topic_distribution(
            best_model,
            suffix="_best"
        )
        best_model.save(str(self.lda_model_path))

        return best_model
    

if __name__=="__main__":
    from topic_modeling.utils.common_utils import read_config
    from topic_modeling.pipeline.corpus_reader import TextsReader
    from topic_modeling.pipeline.lemmatizer import Lemmatizer

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

    lda_topic_modeling = LDATopicModeling(config, lemmatized_texts)
    best_model = lda_topic_modeling()
    print(type(best_model))