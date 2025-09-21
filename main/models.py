import pandas as pd
import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report,accuracy_score
from sklearn.utils.validation import validate_data

from utils import get_data

import warnings


class GallstoneModel():
    def __init__(self, model):
        self.pipe = Pipeline([
            ('scalar', StandardScaler()),
            ('model', model),
        ])
 
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.pipe.fit(X, y)
    
    def predict(self, Xtest):
        return self.pipe.predict(Xtest)
    
    def metrics(self, Xtest, Ytest):
        Ypred = self.pipe.predict(Xtest)
        return classification_report(Ytest, Ypred)
    

X, y, df = get_data()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# supressed labelling warning
warnings.filterwarnings('ignore')

# train and evaluate datasaet
pipe_rf = GallstoneModel(RandomForestClassifier())
pipe_rf.fit(Xtrain, Ytrain)
pipe_rf.metrics(Xtest, Ytest)

pipe_dt = GallstoneModel(tree.DecisionTreeClassifier())
pipe_dt.fit(Xtrain, Ytrain)
pipe_dt.metrics(Xtest, Ytest)

pipe_reg = GallstoneModel(LogisticRegressionCV())
pipe_reg.fit(Xtrain, Ytrain)
pipe_reg.metrics(Xtest, Ytest)

pipe_mlp = GallstoneModel(MLPClassifier())
pipe_mlp.fit(Xtrain, Ytrain)
pipe_mlp.metrics(Xtest, Ytest)

pipe_xgb = GallstoneModel(XGBClassifier())
pipe_xgb.fit(Xtrain, Ytrain)
pipe_xgb.metrics(Xtest, Ytest)

print(pipe_xgb.metrics(Xtest, Ytest))

















         




