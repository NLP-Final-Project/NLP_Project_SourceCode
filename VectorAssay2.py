import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


# Importing the dataset
dataset = pd.read_csv('ProjectSamples.csv') 

# Cleaning the text

attributeList = []
#attributeList.append('project_essay_1')
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
    
    
#Creating the bag of words model

cv = CountVectorizer()
for item in corpus_main:
    X = cv.fit_transform(item).toarray()

df=pd.DataFrame(X)
df.to_csv('Vector_Assay_2.csv')

