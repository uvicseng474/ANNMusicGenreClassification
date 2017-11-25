# -*- coding: cp1252 -*-
from dataUtility import csV2Cla
import os.path
import itertools
import matplotlib.pyplot as plt
import numpy as np
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
        # Here, s(p) = s1(p), s2(p),�, si(p),�, sn(p)
        # Weight Matrix is given by
        
    def test(self, X, y):
        print('todo')


#if not os.path.exists('data/results.pickle'):
#    data = csV2Cla('data/lyrics.csv','data/result.pickle')
#else:
#    data = csV2Cla()
#    data.load('data/result.pickle')

def plot_confusion_matrix(confusion_m, classes, normalize=False, title='Confusion matrix', cmap = plt.cm.Blues):
    
    if normalize:
        confusion_m = confusion_m.astype('float') / confusion_m.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(confusion_m, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation = 45)
    plt.yticks(ticks, classes)

    fmt = 'd' if not normalize else '.2f'
    threshold = confusion_m.max()/2.
    for i, j in itertools.product(range(confusion_m.shape[0]), range(confusion_m.shape[1])):
        plt.text(j,i, format(confusion_m[i,j], fmt), horizontalalignment="center", color="black" if threshold > confusion_m[i,j] else "white")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_classification_report(cr, classes ,title='Classification report ', cmap=plt.cm.Blues):

    lines = cr.split("\n")

    mat = []

    for line in lines[2 : (len(lines) - 3)]:
        t= line.split()
        mat.append([float(x) for x in t[1:len(t)-1]])
    
    aveTotal = lines[len(lines) - 1].split()
    classes.append('avg/total')
    mat.append([float(x) for x in t[1:len(aveTotal) - 1]])

    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick = np.arange(3)
    y_tick = np.arange(len(classes))
    plt.tight_layout()
    plt.xticks(x_tick, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')

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
    
    mlp = MLPClassifier(hidden_layer_sizes=(numAttr))
    print('fitting data')
    mlp.fit(X_train,y_train)
    print('making predictions')
    predictions = mlp.predict(X_test)
    #print (confusion_matrix(y_test,predictions))
    #print (classification_report(y_test,predictions,labels=["Country","Electronic","Folk","Hip-Hop","Indie",
    classes = ["Country","Electronic","Folk","Hip-Hop","Indie", "Jazz","Metal","Pop","R&B","Rock"]
    plt.figure()
    confusion_m = confusion_matrix(y_test,predictions)
    print(confusion_m)
    plot_confusion_matrix(confusion_m, classes)
    plt.figure()
    plot_confusion_matrix(confusion_m, classes, True)
    plt.figure()
    cr = classification_report(y_test,predictions,labels=classes)
    print(cr)
    plot_classification_report(cr,classes)
    plt.show()

# run python neuralNetwork.py
# below statements are to understand data structure
#print("List of genres \n", data.genre)
#print("\nList of years :\n", data.year)
#print("\nList of artists :\n", data.artist)
#print("\nDescription of data :\n", data.base.head())
#print("\nList of lyrics :\n", data.base.lyrics)
