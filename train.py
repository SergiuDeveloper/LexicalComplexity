#!/usr/bin/python3

from utils import *
from constants import *


train_df, test_df = load_datasets(TRAIN_DATASET_PATH, TEST_DATASET_PATH)

train_sentences = list(train_df['sentence'])
train_tokenized_sentences = preprocess_data(train_sentences, train_df['token'], LANGUAGE)
test_sentences = list(test_df['sentence'])
test_tokenized_sentences = preprocess_data(test_sentences, test_df['token'], LANGUAGE)

cbow_model = load_model(CBOW_MODEL_PATH)
if cbow_model == None:
    cbow_model = create_model(False, train_tokenized_sentences, CBOW_MODEL_PATH)
sg_model = load_model(SG_MODEL_PATH)
if sg_model == None:
    sg_model = create_model(True, train_tokenized_sentences, SG_MODEL_PATH)

word_complexities = []
cbow_squared_errors = []
sg_squared_errors = []
for _, test_df_entry in test_df.iterrows():
    test_word = test_df_entry['token']    
    test_complexity = test_df_entry['complexity']

    computed_complexity_cbow = compute_word_complexity(cbow_model, train_df, test_word)
    computed_complexity_sg = compute_word_complexity(sg_model, train_df, test_word)

    word_complexities.append((test_complexity, computed_complexity_cbow, computed_complexity_sg))
    cbow_squared_errors.append((computed_complexity_cbow - test_complexity) ** 2)
    sg_squared_errors.append((computed_complexity_sg - test_complexity) ** 2)

    print('Word: \"{}\"'.format(test_word))
    print('Complexity:', test_complexity)
    print('CBOW computed complexity:', computed_complexity_cbow)
    print('SG computed complexity:', computed_complexity_sg)
    print()

cbow_MSE = sum(cbow_squared_errors) / len(cbow_squared_errors)
sg_MSE = sum(sg_squared_errors) / len(sg_squared_errors)

print('CBOW MSE:', cbow_MSE)
print('SG MSE:', sg_MSE)