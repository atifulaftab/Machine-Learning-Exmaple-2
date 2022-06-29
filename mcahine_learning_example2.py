# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1gnAvchjQquFhxWBdoN1T2WOk1eW4W4_E
"""

from collections import defaultdict
import csv
from itertools import *
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

amino_acid=[]
assigned_class=[]
ac_frequency = {}
cl_frequency = {}
i_freq = {}
ma_freq = {}
va_freq = {}
features = []

def data_reading(data):
    data=data
    dataset=open(data)
    return dataset

def amino_acid_frequecy_idnetification(dataset):
    dataset=dataset
    for data in dataset:
        data=data.strip().split(',')
        if(data[0]!='ID'):
            amino_acid.append(data[1])
            if data[2] == "very active":
                assigned_class.append('1')
            elif data[2] == "mod. active":
                assigned_class.append('2')
            else:
                assigned_class.append('0')
    for aa in amino_acid:
        for aas in aa:
            if aas in ac_frequency:
                ac_frequency[aas] += 1
            else:
                ac_frequency[aas] = 1
    print ("Count of all amino acid in sequence is :\n "+  str(ac_frequency))
    print("\n")
    amino = list(ac_frequency.keys())
    freq = list(ac_frequency.values())
    plt.bar(range(len(amino)), freq, tick_label=amino)
    plt.title("Count of all amino acid in sequence")
    plt.show()
    print("\n")
    a_c=np.array(assigned_class)
    moderately_active=np.sum(a_c=='2')
    very_active=np.sum(a_c=='1')
    inactive=np.sum(a_c
                    =='0')
    class_data=[moderately_active,very_active,inactive]
    labels=['Moderately Active','Very Active','Inactive']
    fig = plt.figure()
    plt.bar(labels,class_data)
    plt.xlabel('Classes')
    plt.ylabel('No of instances')
    plt.title('Class Distribution')
    plt.show()

def data_processing(dataset):
    print("Processing Data\n")
    dataset=dataset
    temp=''
    features=[]
    for data in dataset:
        data=data.strip().split(',')
        if(data[0]!='ID'):
            amino_acid.append(data[1])
        
    for aa in amino_acid:
        for aas in aa:
            if aas in ac_frequency:
                ac_frequency[aas] += 1
            else:
                ac_frequency[aas] = 1
    temp_freq={}
    g = globals()
    for i in ac_frequency.keys():
        g[i] = []
    for aa in amino_acid:
        length= len(aa)
        for aas in aa:
            if aas in temp_freq.keys():
                temp_freq[aas] += 1
            else:
                temp_freq[aas] = 1
        for i in ac_frequency.keys():
            if i in temp_freq.keys():
                temp=temp+str(temp_freq[i]/length)+' '
            else:
                temp=temp+'0'+' '
        features.append(temp)
        temp_freq={} 
        length=0
        temp=''
    #processed_data=np.array([[float(i) for i in j[1:-1].split()] for j in features])
    #print(np.array([[float(i) for i in j[1:-1].split()] for j in features]))
    processed_data=features
    return(processed_data)

def data_split(X,y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    lst_accu_stratified = []
    tr_index=[]
    tst_index=[]
    for train_index, test_index in skf.split(X, y):
        tr_index.append(train_index)
        tst_index.append(test_index)

    return tr_index,tst_index

def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #print("Accuracy for Decision Tree:",metrics.accuracy_score(y_test, y_pred))
    return y_test,y_pred

def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #print("Accuracy for Random Forest:",metrics.accuracy_score(y_test, y_pred))
    return y_test,y_pred

print("Reading Dataset")
dataset=data_reading("ACPs_Breast_Cancer.csv")
amino_acid_frequecy_idnetification(dataset)

processed_data=data_processing(dataset)
assigned_class_converted=np.array(assigned_class)
data=np.zeros((949,22))
for c,i in enumerate(processed_data):
    i=i.split()
    for c2,j in enumerate(i):
        data[c][c2]=float(j)
#[X_train, X_test, y_train, y_test]=data_split(data,assigned_class,TS)

train_index,test_index=data_split(data,assigned_class_converted)

y_test=[]
y_pred=[]
for i in range(5):
    tr_data,tr_label=data[train_index[i]],assigned_class_converted[train_index[i]]
    tst_data,tst_label=data[test_index[i]],assigned_class_converted[test_index[i]]
    y_t,y_p=decision_tree(tr_data,tst_data,tr_label,tst_label)
    y_test=y_test+list(y_t)
    y_pred=y_pred+list(y_p)
print(classification_report(y_test, y_pred))

y_test=[]
y_pred=[]
for i in range(5):
    tr_data,tr_label=data[train_index[i]],assigned_class_converted[train_index[i]]
    tst_data,tst_label=data[test_index[i]],assigned_class_converted[test_index[i]]
    y_t,y_p=random_forest(tr_data,tst_data,tr_label,tst_label)
    y_test=y_test+list(y_t)
    y_pred=y_pred+list(y_p)
print(classification_report(y_test, y_pred))
