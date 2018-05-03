from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from statistics import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import sklearn.ensemble
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


#Choose assay 1 or assay 2 for analyzing
assayNumber = input('Choose which assay that you want to analyze (1 or 2): ')

#Read vector data of assay1
assayData = pd.read_csv('Vector_Assay_'+str(assayNumber)+'.csv')

dataset = pd.read_csv('ProjectSamples.csv')


trainingNumber=input('How many samples do you want to use for training (100-7000): ')
tN=int(trainingNumber)

X_train = assayData.iloc[0:(tN-1),:].values
X_test = assayData.iloc[7000:9999,:].values

y_t = dataset.iloc[0: (tN-1), 15].values
y_test = dataset.iloc[7000: 9999, 15].values


#In SVM model, using crossvalidation and try different C and kernel 
print('In SVM model:')

for n in range (1,5):
    for p in range (0,2):
        s=['linear','rbf']
        clf=SVC(kernel=s[p], C=0.1/(10**n),random_state=0)
        roc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='roc_auc')
        acc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10)
        fscore=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='f1')
        print("C=%0.8f; kernel=%s; ACC=%0.3f(+/- %0.3f); f1=%0.3f(+/- %0.3f)"%((0.1/(10**n)),s[p],acc.mean(),acc.std(),fscore.mean(),fscore.std()))

print()
print()


#In LR model, using crossvalidation and try different C and penalty 
print('In LR model:')

for n in range (1,5):
    for p in range (1,3):
        s='l'+str(p)
        clf=LogisticRegression(C=1.0/(10**n),penalty=s,random_state=0)
        roc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='roc_auc')
        acc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10)
        fscore=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='f1')
        print("C=%0.8f; penalty=%d; ACC=%0.3f(+/- %0.3f); f1=%0.3f(+/- %0.3f)"%((1.0/(10**n)),p,acc.mean(),acc.std(),fscore.mean(),fscore.std()))

print()
print()


#In SGD model, using crossvalidation and try different alpha and penalty 
print('In SGD model:')

for n in range (1,5):
    for p in range (1,3):
        s='l'+str(p)
        clf=linear_model.SGDClassifier(penalty=s, alpha=1.0*(10**n))
        roc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='roc_auc')
        acc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10)
        fscore=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='f1')
        print("alpha=%0.8f; penalty=%d; ACC=%0.3f(+/- %0.3f); f1=%0.3f(+/- %0.3f)"%((1.0*(10**n)),p,acc.mean(),acc.std(),fscore.mean(),fscore.std()))

print()
print()


#In DecisionTree model, using crossvalidation 
print('In DecisionTree model:')

clf=DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
roc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='roc_auc')
acc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10)
fscore=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='f1')
print("ACC=%0.3f(+/- %0.3f); f1=%0.3f(+/- %0.3f)"%(acc.mean(),acc.std(),fscore.mean(),fscore.std()))

print()
print()


#In AdaBoost model, using crossvalidation and try different learning rate 
print('In AdaBoost model:')

for n in range (1,4):
    clf=AdaBoostClassifier(n_estimators=100,learning_rate=1.0/(5**n),random_state=0)
    roc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='roc_auc')
    acc=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10)
    fscore=cross_val_score(estimator=clf,X=X_train,y=y_t,cv=10,scoring='f1')
    print("learning_rate=%0.8f; ACC=%0.3f(+/- %0.3f); f1=%0.3f(+/- %0.3f)"%(1.0/(5**n),acc.mean(),acc.std(),fscore.mean(),fscore.std()))

print()
print()



