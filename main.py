# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:01:52 2023

@author: giosp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from metrics import show_metrics
from imblearn.over_sampling import SMOTE
plt.style.use(['science', 'ieee'])

#Options
balance_data = False

data_directory = 'D:/'
data = pd.read_csv(data_directory + 'creditcard.csv' )

#Print Data Information
print('Shape of the data is ' + str(data.shape))
print('Columns of the data are ' + str(data.columns))
print('Number of Cases of Fraud is ' + str(len(data[data['Class']==1])))
print('Classes Balance is Regular: ' + str(len(data[data['Class']==0])/len(data))
      + ' Fraud: ' + str(len(data[data['Class']==1])/len(data)))

#Study Correlation Among Features
plt.figure(figsize=(10,8))
corr=data.corr()
plt.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()

#%%Split Data into features and Target and Traint-Test Splits
X=data.drop(['Class'],axis=1)
y=data['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1)

#Balance training data 
if balance_data:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    #Print Resabpled Data Information
    print('Classes Balance is Regular: ' + str(len(y_train[y_train==0])/len(y_train))
          + ' Fraud: ' + str(len(y_train[y_train==1])/len(y_train)))

#%% RandomForestClassifier
rfc=RandomForestClassifier()
model=rfc.fit(X_train,y_train)
prediction=model.predict(X_test)

print('Random Forest Metrics are :')
show_metrics(y_test, prediction)

#Logistic Regression
lr=LogisticRegression()
model_lr=lr.fit(X_train,y_train)
prediction=model_lr.predict(X_test)

print('Logisitc Regression Metrics are :')
show_metrics(y_test, prediction)

#Decision Tree Regressor
dt=DecisionTreeRegressor()
model_dt=dt.fit(X_train,y_train)
prediction=model_dt.predict(X_test)

print('Decision Tree Regressor Metrics are :')
show_metrics(y_test, prediction)



