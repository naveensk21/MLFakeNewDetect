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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import pickle


dataset = pd.read_csv('dataset/cleaned_dataset.csv')

# -----------------------------------check ----------------------------------
del dataset['Unnamed: 0']
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
dataset['title_text'] = dataset['title'] + " " + dataset['text']
# ----------------------------------------------------------------------------


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
model = LogisticRegression()

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

# ---------------  ROC Curve ---------------
# get models predicted probabilities
model_probs = model.predict_proba(X_test)
model_probs = model_probs[:, 1]

# get the false positive and true positive rates
fpr, tpr, thresholds = roc_curve(y_test, model_probs, pos_label=1)

# get auc score
roc_auc = roc_auc_score(y_test, model_probs)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# roc curve for tpr = fpr
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

exit()
# --------------- Save the model and vectorizer ---------------
pickle.dump(model, open(f'model/{model.__class__}_model.sav', 'wb'))

pickle.dump(tfidf, open(f"model/{model.__class__}_vectorizer.pickle", "wb"))


