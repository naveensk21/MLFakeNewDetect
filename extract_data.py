import re
import string

import pandas as pd
import opendatasets as od

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem import WordNetLemmatizer

# download the dataset
dataset_link = 'https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification'
od.download(dataset_link)

# load the dataset
dataset = pd.read_csv('fake-news-classification/WELFake_Dataset.csv')

del dataset['Unnamed: 0']

# ------------------ drop na values ------------------
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

# ------------------ add text length -----------------
dataset['Text_length'] = dataset.text.str.split().str.len()

# ------------------ combine text and title ------------------
dataset['title_text'] = dataset['title'] + " " + dataset['text']

# ------------------ preprocess text -----------------
stop_words = set(stopwords.words('english'))
le = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(re.compile('<.*?>'), '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('\d+', '', text)
    text = text.replace("'", "")
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace("’", "")
    text = text.replace("'", "")
    text = text.strip("\'")

    word_tokens = text.split()
    word_tokens = [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    cleaned_text = " ".join(word_tokens)
    cleaned_text = re.sub(r"\b[a-zA-Z]\b", "", cleaned_text)
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text


dataset['title_text'] = dataset['title_text'].apply(preprocess_text)

# ------------------ saved the dataset ------------------
dataset.to_csv('dataset/cleaned_dataset.csv')
