# UNG_topic_modeling
Topic Modeling for the UNG Corpus of Slovenian Literary Texts

## Repository structure

```
.
├── corpus/
├── metadata/
├── notebooks/
├── topic_modeling/  # topic modeling module
│   ├── config/
│   ├── pipeline/
│   │   ├── corpus_reader.py  # reads corpus
│   │   ├── lda_topic_modeling.py  # LDA topic modeling
│   │   └── lemmatizer.py  # lemmatizes corpus
│   ├── stopwords/  # additional stopwords
│   ├── utils/
│   └── pipeline_runner.py  # entry point
```

## Before running the pipeline

Please configure the `topic_modeling/config/config.yaml` before running the pipeline.

Parameters:
- input_data – paths to corpus directory, metadata file and additional stopwords list;
- output_data – names of the output directory, lemmatized corpus directory, LDA model etc;
- LDA parameters:
    - limit – maximum number of topics;
    - start – starting number of topics;
    - step – number of topic steps for training the new topic model;
    - num_of_topic_words – number of words per topic;
    - num_of_topics_per_document – number of topics per single document;
- num_of_texts_to_read – number of texts used for topic modeling.

## Entry point

To run topic modeling, execute the following command from the repository root:

```bash
python -m topic_modeling.pipeline_runner
```