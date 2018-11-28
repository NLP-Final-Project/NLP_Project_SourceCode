# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:05:09 2018

@author: Rohit
"""

# Importing the Libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Input_File.csv') # quoting ==> ignore double quotes

# Cleaning the text
import re
import nltk

# Import NLTK libraries to remove stopwords and lemmatize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Fetching the essay attributes
attributeList = []
attributeList.append('project_essay_1')
attributeList.append('project_essay_2')

# preprocess/clean the essay attributes by removing stop words and perform lemmatization
corpus_main = []
for attr in attributeList:
    corpus = []
    for i in range(0, 1000):
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
    
# convering the essay attributes to bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
for item in corpus_main:
    X = cv.fit_transform(item).toarray()
    
y = dataset.iloc[0: 1000, 15].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Importing classification model librares
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Importing confusion matrix library
from sklearn.metrics import confusion_matrix

# map the inputs to the function blocks
options = {0 : "Logistic_Regression",
           1 : "K_Nearest_Neighbors",
           2 : "Naive_Bayes",
           3 : "Decision_Tree",
           4 : "Random_Forest"
          }

# function blocks for training and predicting the scores
def Logistic_Regression():
    classifierL = LogisticRegression(random_state = 0)
    classifierL.fit(X_train, y_train)
    y_predL = classifierL.predict(X_test)
    cmL = confusion_matrix(y_test, y_predL)
    accuracyL = (cmL[0][0] + cmL[1][1]) / (cmL[0][0] + cmL[0][1] + cmL[1][0] + cmL[1][1])
    return accuracyL

def K_Nearest_Neighbors():
    classifierK = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)    
    classifierK.fit(X_train, y_train)
    y_predK = classifierK.predict(X_test)
    cmK = confusion_matrix(y_test, y_predK)
    accuracyK = (cmK[0][0] + cmK[1][1]) / (cmK[0][0] + cmK[0][1] + cmK[1][0] + cmK[1][1])
    return accuracyK
    
def Naive_Bayes():
    classifierN = GaussianNB()
    classifierN.fit(X_train, y_train)
    y_predN = classifierN.predict(X_test)
    cmN = confusion_matrix(y_test, y_predN)
    accuracyN = (cmN[0][0] + cmN[1][1]) / (cmN[0][0] + cmN[0][1] + cmN[1][0] + cmN[1][1])
    return accuracyN
    
def Decision_Tree():
    classifierD = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifierD.fit(X_train, y_train)
    y_predD = classifierD.predict(X_test)
    cmD = confusion_matrix(y_test, y_predD)
    accuracyD = (cmD[0][0] + cmD[1][1]) / (cmD[0][0] + cmD[0][1] + cmD[1][0] + cmD[1][1])
    return accuracyD

def Random_Forest():
    classifierR = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifierR.fit(X_train, y_train)
    y_predR = classifierR.predict(X_test)
    cmR = confusion_matrix(y_test, y_predR)
    accuracyR = (cmR[0][0] + cmR[1][1]) / (cmR[0][0] + cmR[0][1] + cmR[1][0] + cmR[1][1])
    return accuracyR

# Loop through all models and generate the accuracy of each model
for model in range(5):
    if(model == 0):
        print("Accuracy of Logistic_Regression: ",  Logistic_Regression())
    elif(model == 1):
        print("Accuracy of K_Nearest_Neighbors: ", K_Nearest_Neighbors())
    elif(model == 2):
        print("Accuracy of Naive_Bayes: ", Naive_Bayes())
    elif(model == 3):
        print("Accuracy of Decision_Tree: ", Decision_Tree())
    else:
        print("Accuracy of Random_Forest: ", Random_Forest())
        

#######################################################################################################
# Artificial Neural Network  
#######################################################################################################        
        
# Import ANN libraries   
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifierANN = Sequential()

# Adding the input layer and the first hidden layer
classifierANN.add(Dense(output_dim = 1671, init = 'uniform', activation = 'relu', input_dim = 3341))

# Adding the second hidden layer
classifierANN.add(Dense(output_dim = 1671, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifierANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifierANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifierANN.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

# Predicting the Test set results
y_predANN = classifierANN.predict(X_test)
y_predANN = (y_predANN > 0.5)

# Confusion Matrix
cmANN = confusion_matrix(y_test, y_predANN)

accuracyANN = (cmANN[0][0] + cmANN[1][1]) / (cmANN[0][0] + cmANN[0][1] + cmANN[1][0] + cmANN[1][1])

