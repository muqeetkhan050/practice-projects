# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:47:25 2022

@author: Muqeet
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
data=pd.read_csv('attrition_data.csv')

data.describe()
data.info()
data.shape

np.random.seed(37)

print('shape of data',data.shape)

data.isnull().sum()

#dropping irrelevant columns 
data.drop(['EMP_ID','JOBCODE','TERMINATION_YEAR'],axis=1,inplace=True)
data.drop(data.iloc[:,-5:],axis=1,inplace=True)
data.isnull().sum()

data['REFERRAL_SOURCE'].fillna(data['REFERRAL_SOURCE'].mode()[0],inplace=True)

data['STATUS'].unique()


#visualization

sns.set(style='darkgrid')
a=sns.countplot(x='STATUS',data=data)
plt.xlabel('status')
plt.ylabel('count')
plt.show()

data['AGE'].unique()

data['STATUS'].unique()

plt.figure(figsize=(20,20))
for x in ['T','A']:
    data['AGE'][data['STATUS']==x].plot(kind='kde')

plt.title("status vs age density")
plt.legend(('T','A'))
plt.xlabel('Age')
plt.show()

data['PERFORMANCE_RATING'].unique()

sns.countplot(x='PERFORMANCE_RATING',hue='STATUS',data=data)
plt.show()   


#feature engineering

data['NUMBER_OF_TEAM_CHANGED'].unique()

le=LabelEncoder()
data['NUMBER_OF_TEAM_CHANGED']=le.fit_transform(data['NUMBER_OF_TEAM_CHANGED'])
data['REHIRE']=le.fit_transform(data['REHIRE'])
data['IS_FIRST_JOB']=le.fit_transform(data['IS_FIRST_JOB'])
data['IS_FIRST_JOB'] = le.fit_transform(data['IS_FIRST_JOB'])
data['TRAVELLED_REQUIRED'] = le.fit_transform(data['TRAVELLED_REQUIRED'])
data['DISABLED_EMP'] = le.fit_transform(data['DISABLED_EMP'])
data['DISABLED_VET'] = le.fit_transform(data['DISABLED_VET'])
data['EDUCATION_LEVEL'] = le.fit_transform(data['EDUCATION_LEVEL'])
data['STATUS'] = le.fit_transform(data['STATUS'])


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True)
plt.show()

data.drop(['HRLY_RATE'],axis=1,inplace=True)

#one hot encoding
data['HIRE_MONTH']=data['HIRE_MONTH'].astype('category')
data['JOB_GROUP']=data['JOB_GROUP'].astype('category')
data['REFERRAL_SOURCE']=data['REFERRAL_SOURCE'].astype('category')
data['ETHNICITY']=data['ETHNICITY'].astype('category')
data['SEX']=data['SEX'].astype('category')
data['MARITAL_STATUS'] = data['MARITAL_STATUS'].astype('category')
data=pd.get_dummies(data,columns=['MARITAL_STATUS','SEX','HIRE_MONTH','JOB_GROUP','REFERRAL_SOURCE','ETHNICITY'])


X=data.drop(['STATUS'],axis=1)
y=data['STATUS']


scaler=StandardScaler()
X=scaler.fit_transform(X)


#logisticregression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

cv=StratifiedShuffleSplit(n_splits=10,test_size=.30,random_state=15)

# Building our model with K-fold validation and GridSearch to find the best parameters

# Defining all the parameters
params = {
    'penalty': ['l1','l2'],
    'C': [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
}

# Building model
logreg = LogisticRegression(solver='liblinear')

# Parameter estimating using GridSearch
grid = GridSearchCV(logreg, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)

# Fitting the model
grid.fit(X_train, y_train)



print('Best score',grid.best_score_)
print('Best Params:', grid.best_params_)
print('Best Estimator:', grid.best_estimator_)

log_reg_grid=grid.best_estimator_
y_pred=log_reg_grid.predict(X_test)


#K_nearest neighbors classsifier(KNN)

parameters={
    'n_neighbors':[3,4,11,19],
    'weights':['uniform','distance']
    }

from sklearn.neighbors import KNeighborsClassifier


knn=KNeighborsClassifier()
cv=StratifiedShuffleSplit(n_splits=10,test_size=0.30,random_state=15)
grid=GridSearchCV(knn,param_grid=parameters,scoring='accuracy',n_jobs=-1,cv=cv)


grid.fit(X_train,y_train)


print('best score:',grid.best_score_)
print('best estiimator:',grid.best_estimator_)
print('best params:',grid.best_params_)

#confusion matrix
knn_grid=grid.best_estimator_
y_pred=knn_grid.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred),columns=['predicted A','Predicted T'])
classification_report(y_test, y_pred)


#Decision tree
# Building our model with K-fold validation and GridSearch to find the best parameters

from sklearn.tree import DecisionTreeClassifier
cv=StratifiedShuffleSplit(n_splits=10,test_size=30,random_state=15)
dts=DecisionTreeClassifier()
params={
    'max_features':[1,3,10],
    'min_samples_split':[2,3,10],
    'min_samples_leaf':[1,3,10],
    'criterion':['entropy','gini']
    }


grid=GridSearchCV(dts,param_grid=params,scoring='accuracy',n_jobs=-1,cv=cv)

grid.fit(X_train,y_train)

print('best score:',grid.best_score_)
print('best eliminator',grid.best_estimator_)
print('best params:',grid.best_params_)



dtc_grid=grid.best_estimator_
y_predict=dtc_grid.predict(X_test)

pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"])




















