# Project Overview
The rise of social media has facilitated the widespread spread of false information, with platforms like Twitter, Facebook, and Reddit being particularly susceptible to misinformation. This has led to significant challenges for society, including the dissemination of rumors, manipulation of political outcomes, and the increase in clickbait content. Detecting fake news is essential for maintaining societal well-being, and various methods, including traditional classification and advanced neural networks, have been developed to address this issue.

The project aims to develop a natural language processing application from the ground up and then host it using Flask.

# Python Packages Used
General Purpose: os, request, re, json, time <br /> 
Data Manipulation: pandas, NumPy <br /> 
Data Cleaning: [NLTK](https://www.nltk.org/) <br /> 
Data Visualization: [Matplotlib](https://matplotlib.org/)<br /> 
Machine Learning: [Scikit-learn](https://scikit-learn.org/stable/install.html), [Tensorflow](https://www.tensorflow.org/guide/keras), [Gensim](https://pypi.org/project/gensim/), [Optuna](https://optuna.org/) 

# Dataset
(WELFake) is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news.
The dataset contains four columns: Serial number (starting from 0); Title (about the text news heading); Text (about the news content); and Label (0 = fake and 1 = real).
[WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

# Results 

## ML/DL Model Results

| Models   | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| RF       |   0.94    |  0.94  |   0.94   |
| SVM      |   0.96    |  0.97  |   0.96   |
| LR       |   0.94    |  0.95  |   0.95   |
| LSTM     |   0.97    |  0.98  |    -     |

## Evaluation
### LSTM Model 

|![](figures/LSTM_Loss.png)<br>LSTM Loss History|![](figures/LSTM_Precision.png)<br>LSTM Precision History|![](figures/LSTM_Recall.png)<br>LSTM Recall History|
|:-:|:-:|:-:|

### Random Forest and Logistic Regression 
|![](figures/RF_ROC.png)<br>RF ROC|![](figures/LR_ROC.png)<br>LR ROC|
|:-:|:-:|


