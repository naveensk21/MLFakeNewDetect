import pandas as pd
import json
import csv
import sys
import re
import string
import time
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from gensim.models import Word2Vec

from keras import layers
from tensorflow import keras
from keras.layers import Dense, Activation, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Conv2D, LSTM, Flatten, BatchNormalization, Bidirectional
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.utils import compute_sample_weight

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


# convert data to json
csv_data = 'dataset/cleaned_dataset.csv'
csv.field_size_limit(sys.maxsize)

with open(csv_data, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

with open('dataset/clean_dataset.json', 'w') as jsonfile:
    json.dump(data, jsonfile)

# read dataset
data_filepath = 'dataset/clean_dataset.json'

with open(data_filepath) as fp:
    dataset_json = json.load(fp)


# get x and y values
def get_x_y():
    X = []
    y = []

    for datapoint in dataset_json:
        y.append(datapoint['label'])
        X.append(datapoint['title_text'])

    return X, y


X, y = get_x_y()

# change the y values from str to int
y = [eval(i) for i in y]

# ----------------- preprocess the text -----------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(re.compile('<.*?>'), '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('\d+', '', text)
    text = re.sub(r"\bhttp\w+", "", text) # remove words that start with http

    word_tokens = text.split()
    le=WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    word_tokens = [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    cleaned_text = " ".join(word_tokens)
    cleaned_text = re.sub(r"\b[a-zA-Z]\b", "", cleaned_text)
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text


preprocessed_text = list(map(lambda text: preprocess_text(text), X))


# ----------------- build the word2vec model -----------------
def create_tokens(sentence_list):
    token_data = []
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        token_data.append(tokens)
    return token_data


tokenized_data = create_tokens(preprocessed_text)

start = time.time()
w2v_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=7, min_count=1)
print("Time taken to train word2vec model: ", round(time.time()-start, 0), 'seconds')

vocab = w2v_model.wv.key_to_index
print("The total words : ", len(vocab))
vocab = list(vocab.keys()) # list of the words in the model


def create_word_vect_dict(vocab):
    wv_dict = {}
    for word in vocab:
        wv_dict[word] = w2v_model.wv.get_vector(word)
    return wv_dict


word_vect_dict = create_word_vect_dict(vocab)

# encode the text and define parameters (tokenize the text)
# fit_on_texts Updates internal vocabulary based on a list of texts.
# This method creates the vocabulary index based on word frequency.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_text)

vocab_size = len(tokenizer.word_index) + 1 #total vocabulary size
encoded_ptext = tokenizer.texts_to_sequences(preprocessed_text) # encoded policy texts with mathematical index
maxlen = 1000 #maximum length news articles
embed_dim = 100

with open('model/lstm_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saved lstm vectorizer')

exit()

# creating the embedding matrix
def get_embedding_matrix():
    embed_matrix = np.zeros(shape=(vocab_size, embed_dim))
    hits = 0
    misses = 0

    for word, i in tokenizer.word_index.items():
        embed_vector = word_vect_dict.get(word)
        try:
            if embed_vector is not None:
                embed_matrix[i] = embed_vector
                hits += 1
            else:
                misses += 1
        except:
            pass

    print(f'converted words {hits} ({misses} missed words)')
    return embed_matrix


embedding_matrix = get_embedding_matrix()

print(embedding_matrix)


# pad the sequence of policy text
def pad_text(text):
    padding = pad_sequences(text, maxlen=maxlen, padding='post')
    return padding


padded_text = pad_text(encoded_ptext)


# define a simple lstm neural network
def lstm_model():
    model = Sequential()

    model.add(Embedding(vocab_size, embed_dim, input_length=maxlen,
                        weights=[embedding_matrix], trainable=False))

    model.add(LSTM(128))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

    return model


lstm_model = lstm_model()
lstm_model.summary()

y_array = np.array(y)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(padded_text, y_array, test_size=0.2)

callback = EarlyStopping(monitor='loss', patience=5)

filepath = 'lstm_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

history = lstm_model.fit(X_train, y_train, epochs=15, batch_size=16,
                         validation_split=0.1,
                         callbacks=[callback, checkpoint], shuffle=True)

lstm_score = lstm_model.evaluate(X_test, y_test)
print(f'{lstm_model.metrics_names[0]}: {lstm_score[0]}')
print(f'{lstm_model.metrics_names[1]}: {lstm_score[1]}')
print(f'{lstm_model.metrics_names[2]}: {lstm_score[2]}')

# save tokenizer
with open('lstm_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('saved keras tokenizer')






