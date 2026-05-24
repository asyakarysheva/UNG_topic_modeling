import os
import chardet
from tqdm import tqdm
from pathlib import Path


class TextsReader:
    """
    Reads texts and identifies texts with the encoding different from utf-8 or iso-8859-2
    """
    def __init__(
            self,
            config: dict
    ):
        self.config = config

        self.corpus_dir = self.config["input_data"]["corpus_dir"]
        self.num_of_texts_to_read = self.config["num_of_texts_to_read"]

        self.output_dir = self.config["output_data"]["output_dir"]
        self.lemmatized_corpus_dir = Path(self.output_dir).joinpath(
            self.config["output_data"]["lemmatized_corpus_dir"]
        )
        os.makedirs(self.lemmatized_corpus_dir, exist_ok=True)
        self.already_lemmatized_files = {
            file.name for file in self.lemmatized_corpus_dir.iterdir() 
            if file.is_file()
        }

    def read_texts(
            self
    ) -> list[dict]:
        """
        Reads texts from the directory taking into account their encodings.

        Parameters:
        - corpus_dir: path to directory containing the corpus    

        Returns:
        - all_texts: dictionary containing the texts from the corpus and their metadata
        """
        # Getting file names
        all_texts_names = sorted(os.listdir(self.corpus_dir))
        
        if isinstance(self.num_of_texts_to_read, int):
            all_texts_names = all_texts_names[:self.num_of_texts_to_read]

        texts_to_read = [
            file for file in all_texts_names 
            if file not in self.already_lemmatized_files
        ]

        all_texts = []

        for text_name in tqdm(texts_to_read,
                              desc="Reading texts...",
                              total=len(texts_to_read)):
            text_path = Path(self.corpus_dir).joinpath(text_name)
            with open(text_path, "rb") as file:
                raw = file.read()
                an_encoding = chardet.detect(raw)

                if an_encoding["encoding"] == "utf-8":
                    current_encoding = an_encoding["encoding"]
                    text = raw.decode(an_encoding["encoding"])
                else:
                    current_encoding = "iso-8859-2"
                    text = raw.decode(current_encoding)

                dict_ = {
                    "file_name": text_name.removesuffix(".txt"),
                    "full_path": text_path,
                    "encoding": current_encoding,
                    "text": text
                }
                all_texts.append(dict_)
                
        return all_texts

    def find_other_encodings(
            self,
            all_texts : list[dict]
    ) -> list[dict]:
        """
        Finds texts with the encodings different from 'utf-8'.

        Parameters:
        - all_texts : dictionary containing the texts from the corpus and their metadata

        Returns:
        - texts_with_other_encodings: texts with the encodings different from 'utf-8'
        """
        texts_with_other_encodings = []
        for text in all_texts:
            if text["encoding"] != "utf-8":
                texts_with_other_encodings.append(text)

        return texts_with_other_encodings
    
    def __call__(self) -> tuple[list[dict], list[dict]]:
        """
        Runs text reader:
        - reads all the texts from the corpus directory;
        - identifies the texts with other encodings.

        Returns:
        - all_texts: dictionary containing the texts from the corpus and their metadata
        - texts_with_other_encodings: texts with the encodings different from 'utf-8'
        """
        all_texts = self.read_texts()
        texts_with_other_encodings = self.find_other_encodings(all_texts)
        return all_texts, texts_with_other_encodings


if __name__=="__main__":
    from topic_modeling.utils.common_utils import read_config

    config = read_config("topic_modeling/config/config.yaml")
    texts_reader = TextsReader(config)
    all_texts, texts_with_other_encodings = texts_reader()

    print(f"The first text in the corpus: {all_texts[0]}")
    print(f"Author and title: {all_texts[0]['file_name']}")
    print(f"Publication year: {all_texts[0]['publication_year']}")

    text_names_with_other_encodings = [
        text["file_name"]
        for text in texts_with_other_encodings
    ]
    print(f"The names of the texts with other encodings: {text_names_with_other_encodings}")
    
    print("All files have been read!")
