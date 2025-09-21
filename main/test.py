import pytest

from models import GallstoneModel
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import get_data

def test_GallstoneModel():
    X, y, df = get_data()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    pipe = GallstoneModel(XGBClassifier())
    pipe.fit(Xtrain,Ytrain)
    Ypred = pipe.predict(Xtest)
    assert Ypred[0] > 0
    metrics = pipe.metrics(Xtest,Ytest)
    assert len(metrics) > 0 








