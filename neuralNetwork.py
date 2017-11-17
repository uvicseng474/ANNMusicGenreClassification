# -*- coding: cp1252 -*-
from dataUtility import csV2Cla
import os.path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd

class NeuralNetwork():
    
    def __init__(self):
        print('todo')
        # store helpful data structures

    def train(self, X, y):
        print('todo')
        # Binary input patterns
        # For a set of binary patterns s(p), p = 1 to P
        # Here, s(p) = s1(p), s2(p),…, si(p),…, sn(p)
        # Weight Matrix is given by
        
    def test(self, X, y):
        print('todo')


#if not os.path.exists('data/results.pickle'):
#    data = csV2Cla('data/lyrics.csv','data/result.pickle')
#else:
#    data = csV2Cla()
#    data.load('data/result.pickle')
if __name__ == '__main__':

    if not os.path.exists('data/preprocessed_data.csv'):
        sys.exit("file preprocessed_data.csv not found")

    data = pd.read_csv('data/preprocessed_data.csv', sep=',', dtype = None)
    X = data.values[:,3:]
    X = X.astype(int)
    y = data.values[:,1]
    y = y.astype(str)

    numAttr = len(X[0])
    numItems = len(y)


    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    mlp = MLPClassifier(hidden_layer_sizes=(numAttr,numAttr,numAttr))
    print('fitting data')
    mlp.fit(X_train,y_train)
    print('making predictions')
    predictions = mlp.predict(X_test)
    print (confusion_matrix(y_test,predictions))
    print (classification_report(y_test,predictions,labels=["Country","Electronic","Folk","Hip-Hop","Indie",
                                                            "Jazz","Metal","Pop","R&B","Rock"]))

# run python neuralNetwork.py
# below statements are to understand data structure
#print("List of genres \n", data.genre)
#print("\nList of years :\n", data.year)
#print("\nList of artists :\n", data.artist)
#print("\nDescription of data :\n", data.base.head())
#print("\nList of lyrics :\n", data.base.lyrics)
