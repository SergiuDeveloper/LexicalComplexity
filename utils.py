#!/usr/bin/python3

from gensim.models import FastText
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import re


def load_datasets(train_dataset_path, test_dataset_path):
    train_df = pd.read_csv(train_dataset_path, sep='\t')
    test_df = pd.read_csv(test_dataset_path, sep='\t')

    return train_df, test_df

def preprocess_data(sentences, protected_words, language):
    language_stopwords = set(stopwords.words(language))

    sentences = [re.sub(r'[^a-zA-Z\s]', '', sentence).lower() for sentence in sentences]
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [
        [word for word in sentence if word in protected_words or word not in language_stopwords]
        for sentence in tokenized_sentences
    ]

    return tokenized_sentences

def load_model(model_path):
    try:
        model = FastText.load(model_path)
    except:
        model = None
    
    return model

def create_model(skip_gram, tokenized_sentences, model_path):
    model = FastText(min_count=1, window=5, sg=skip_gram)
    model.build_vocab(sentences=tokenized_sentences)
    model.train(sentences=tokenized_sentences, total_examples=len(tokenized_sentences), vector_size=5, epochs=100)

    model.save(model_path)

    return model

def compute_word_complexity(model, train_df, word):
    word = word.lower()

    complexity = 0
    total_similarities_sum = sum([model.wv.similarity(word, train_word) for train_word in model.wv.vocab.keys()])
    for train_word in train_df['token']:
        train_word = str(train_word)

        similarity = model.wv.similarity(word, train_word.lower())
        scaled_similarity = similarity / total_similarities_sum

        train_word_complexity_entries = train_df[train_df['token'] == train_word]['complexity']
        if train_word_complexity_entries.size > 0:
            train_word_complexity = train_word_complexity_entries.mean()
        else:
            train_word_complexity = 0

        complexity += scaled_similarity * train_word_complexity

    return complexity