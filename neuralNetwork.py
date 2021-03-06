# -*- coding: cp1252 -*-
from dataUtility import csV2Cla
import os.path
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


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
    print('Start time: ' + time.strftime('%H:%M:%S'))
    file_path = 'data/preprocessed_data.csv'
    if not os.path.exists(file_path):
        raise Exception("file {} not found".format(file_path))

    data = pd.read_csv(file_path, sep=',', dtype = None)
    X = data.values[:,3:]
    X = X.astype(int)
    y = data.values[:,1]
    y = y.astype(str)

    numAttr = len(X[0])
    numItems = len(y)

    n_trials = 3

    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    for i in range(n_trials):

        print('Trial '+str(i+1))
        X_train, X_test, y_train, y_test = train_test_split(X,y)

        mlp = MLPClassifier(hidden_layer_sizes=(numAttr), activation='logistic', solver='lbfgs', max_iter=500)
        # faster version
        #mlp = MLPClassifier(hidden_layer_sizes=(numAttr), activation='logistic', solver='adam', learning_rate_init=0.1, max_iter=500)
        print('\tTraining...')
        mlp.fit(X_train,y_train)
        print('\tPredicting...')
        predictions = mlp.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average=None)
        recall = recall_score(y_test, predictions, average=None)
        f1 = []
        for i, j in zip(precision, recall):
            if ((i == 0.0) and (j == 0.0)):
                f1.append(0.0)
            else:
                f1.append(2 * i * j / (i + j))

        # f1_score = f1_score(y_test,predictions,average=None)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    accuracy_array = np.asarray(accuracy_list)
    f1_array = np.asarray(f1_list)
    precision_array = np.asarray(precision_list)
    recall_array = np.asarray(recall_list)

    mean_accuracy = np.mean(accuracy_array)
    mean_f1 = np.mean(f1_array,axis=0)
    mean_precision = np.mean(precision_array,axis=0)
    mean_recall = np.mean(recall_array,axis=0)

    std_accuracy = np.std(accuracy_array, ddof=1, axis=0)
    std_f1 = np.std(f1_array, ddof=1, axis=0)
    std_precision = np.std(precision_array, ddof=1, axis=0)
    std_recall = np.std(recall_array, ddof=1, axis=0)


    #Turn 0.0 into np.nan and re-calculate precision,recall,f1
    temp = f1_array
    temp[temp==0.0] = np.nan
    nan_f1_array = temp

    temp = precision_array
    temp[temp==0.0] = np.nan
    nan_precision_array = temp

    temp = recall_array
    temp[temp==0.0] =np.nan
    nan_recall_array = temp

    nan_mean_f1 = np.nanmean(f1_array,axis=0)
    nan_mean_precision = np.nanmean(precision_array,axis=0)
    nan_mean_recall = np.nanmean(recall_array,axis=0)

    nan_std_f1 = np.nanstd(f1_array, ddof=1, axis=0)
    nan_std_precision = np.nanstd(precision_array, ddof=1, axis=0)
    nan_std_recall = np.nanstd(recall_array, ddof=1, axis=0)


    print('Means:')
    print('Accuracy: '+str(mean_accuracy))
    print('F1-Score '+str(mean_f1))
    print('Precision '+str(mean_precision))
    print('Recall '+str(mean_recall))
    print('NAN F1-Score '+str(nan_mean_f1))
    print('NAN Precision '+str(nan_mean_precision))
    print('NAN Recall '+str(nan_mean_recall))

    print('')
    print('Standard Deviations:')
    print('Accuracy: '+str(std_accuracy))
    print('F1-Score '+str(std_f1))
    print('Precision '+str(std_precision))
    print('Recall '+str(std_recall))
    print('NAN F1-Score ' + str(nan_std_f1))
    print('NAN Precision ' + str(nan_std_precision))
    print('NAN Recall ' + str(nan_std_recall))


    a = mean_f1
    b = mean_precision
    c = mean_recall

    labels=["Country","Electronic","Folk","Hip-Hop","Indie","Jazz","Metal","Pop","R&B","Rock"]

    fig = plt.figure(1,figsize=(18.5,10.5),dpi=80)
    ax = plt.subplot(111)
    ind = np.arange(len(labels))
    width = 0.2
    ax.bar(ind-width, a, width, color='b',align='center',label='Mean F1',yerr=std_f1)
    ax.bar(ind, b, width, color='g',align='center',label='Mean Precision',yerr=std_precision)
    ax.bar(ind+width, c, width, color='r',align='center',label='Mean Recall',yerr=std_recall)
    ax.set_xlim(-width,len(ind)+width)
    plt.title("F1-measure, Precision, Recall \nfor each Music Genre", fontsize=22)
    plt.xlabel("Music Genres", fontsize='20')
    plt.ylabel("F1-measure, Precision, Recall", fontsize='20')
    xTickMarks = labels
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation = 25, fontsize = 18)
    plt.tick_params(axis='x',which='major',labelsize='18')
    plt.tick_params(axis='y',which='major',labelsize='18')
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, fontsize=20)
    ax.yaxis.grid()
	
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
    print('End time: ' + time.strftime('%H:%M:%S'))
    plt.show()
