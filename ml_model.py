import pandas as pd

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from sklearn import metrics
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import pickle


dataset = pd.read_csv('dataset/cleaned_dataset.csv')


# --------------- assign x and y values ---------------
X = dataset['title_text']
y = dataset['label']

# --------------- split the dataset ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------- vectorization ---------------
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# --------------- Train the model ---------------
model = RandomForestClassifier()

model.fit(X_train, y_train)

# ---------------  evaluate the model ---------------
y_pred = model.predict(X_test)

accu = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accu)
print('Precision:', prec)
print('Recall:', recall)
print('F1 Score:', f1)

# --------------- Save the model and vectorizer ---------------
pickle.dump(model, open('model/rf_model.sav', 'wb'))

pickle.dump(tfidf, open("model/vectorizer.pickle", "wb"))


