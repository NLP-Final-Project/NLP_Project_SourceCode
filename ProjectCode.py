# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:49:32 2018

@author: Rohit
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Input_File.csv') # quoting ==> ignore double quotes

# Cleaning the text
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

attributeList = []
attributeList.append('project_essay_1')
attributeList.append('project_essay_2')

corpus_main = []
for attr in attributeList:
    corpus = []
    for i in range(0, 10000):
        essay_attribute = attr
        essay = re.sub('[^a-zA-Z]', ' ', str(dataset[essay_attribute][i]))
        essay = essay.lower()
        essay = essay.split()
        ps = PorterStemmer()
        essay = [ps.stem(word) 
        for word in essay if not word in set(stopwords.words('english'))]
        essay = ' '.join(essay)
        corpus.append(essay)
    corpus_main.append(corpus)
    
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
for item in corpus_main:
    X = cv.fit_transform(item).toarray()
    
y = dataset.iloc[0: 10000, 15].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
    