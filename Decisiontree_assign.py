# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:58:23 2021

@author: Amarnadh Tadi
"""

import pandas as pd
import numpy as np
cloth_data=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign9\Company_Data.csv")
##EDA on data
cloth_data.columns
cloth_data.dtypes
cloth_data.isnull().sum()
##Data preprocessing
##Label encoding of catgorical columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cloth_data['ShelveLoc']=le.fit_transform(cloth_data['ShelveLoc'])
cloth_data['Urban']=le.fit_transform(cloth_data['Urban'])
cloth_data['US']=le.fit_transform(cloth_data['US'])


##discretiation of data
# discretization transform the raw data
cloth_data['Sales']=cloth_data['Sales'].astype("category")


cloth_data['Sales'].unique
cloth_data['Sales'].value_counts()
predictors=cloth_data.drop(['Sales'], axis='columns')
predictors.head()
target=cloth_data.iloc[:,[0]]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)

#Decision tree classifier model
#Buliding the model decision tree classifier
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion="entropy")
##Training the model
model.fit(x_train,y_train)

##To test the accuracy
model.score(x_test,y_test)

##checking of first 10 samples of test data
x_test[:10]
y_test[:10]
##prediction of test data
model.predict(x_test[:10])
##prediction of train data
x_train[:10]
y_train[:10]
model.predict(x_train[:10])

##Random forest classifier model
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(n_estimators=15)
model1.fit(x_train,y_train)
model1.score(x_test,y_test)

##plotting confusion matrix
y_predicted=model1.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')

###Diabetes data
import pandas as pd
import numpy as np
diabetes_data=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign9\Diabetes.csv")
##EDA on data
diabetes_data.columns
diabetes_data.dtypes
diabetes_data.isnull().sum()
##Data preprocessing
##Label encoding of catgorical columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
diabetes_data[' Class variable']=le.fit_transform(diabetes_data[' Class variable'])





diabetes_data[' Class variable'].unique
diabetes_data[' Class variable'].value_counts()
predictors=diabetes_data.drop([' Class variable'], axis='columns')
predictors.head()
target=diabetes_data.iloc[:,[8]]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)

#Decision tree classifier model
#Buliding the model decision tree classifier
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion="entropy")
##Training the model
model.fit(x_train,y_train)

##To test the accuracy
model.score(x_test,y_test)

##checking of first 10 samples of test data
x_test[:10]
y_test[:10]
##prediction of test data
model.predict(x_test[:10])
##prediction of train data
x_train[:10]
y_train[:10]
model.predict(x_train[:10])

##Random forest classifier model
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(n_estimators=30)
model1.fit(x_train,y_train)
model1.score(x_test,y_test)

##plotting confusion matrix
y_predicted=model1.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')

##for fraud_check data
import pandas as pd
import numpy as np
fraud_check=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign9\Fraud_check.csv")
fraud_check.columns
fraud_check.dtypes
fraud_check['Taxable.Income']=np.where(fraud_check['Taxable.Income']<=30000,"Risky","Good")
fraud_check['Taxable.Income'].value_counts()


##Label encoding of catgorical columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
fraud_check['Undergrad']=le.fit_transform(fraud_check['Undergrad'])
fraud_check['Marital.Status']=le.fit_transform(fraud_check['Marital.Status'])
fraud_check['Taxable.Income']=le.fit_transform(fraud_check['Taxable.Income'])
fraud_check['Urban']=le.fit_transform(fraud_check['Urban'])



predictors=fraud_check.drop(['Taxable.Income'], axis='columns')
predictors.head()
target=fraud_check.iloc[:,[2]]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)

#Decision tree classifier model
#Buliding the model decision tree classifier
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion="entropy")
##Training the model
model.fit(x_train,y_train)

##To test the accuracy
model.score(x_test,y_test)

##checking of first 10 samples of test data
x_test[:10]
y_test[:10]
##prediction of test data
model.predict(x_test[:10])
##prediction of train data
x_train[:10]
y_train[:10]
model.predict(x_train[:10])

##Random forest classifier model
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(n_estimators=20)
model1.fit(x_train,y_train)
model1.score(x_test,y_test)

##plotting confusion matrix
y_predicted=model1.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')

##for HR data
import pandas as pd
hr_data=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign9\HR_DT.csv")
hr_data.columns
hr_data.dtypes
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
hr_data['Position of the employee']=le.fit_transform(hr_data['Position of the employee'])
hr_data['no of Years of Experience of employee']=hr_data['no of Years of Experience of employee'].astype("category")
hr_data.head()






predictors=hr_data.drop(['Position of the employee'], axis='columns')
predictors.head()
target=hr_data.iloc[:,[0]]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)

#Decision tree classifier model
#Buliding the model decision tree classifier
from sklearn.tree import DecisionTreeClassifier as DT
model=DT()
##Training the model
model.fit(x_train,y_train)
x_train.shape
y_train.shape
##To test the accuracy
model.score(x_test,y_test)

##checking of first 10 samples of test data
x_test[:10]
y_test[:10]
##prediction of test data
model.predict(x_test[:10])
##prediction of train data
x_train[:10]
y_train[:10]
model.predict(x_train[:10])

##Random forest classifier model
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(n_estimators=20)
model1.fit(x_train,y_train)
model1.score(x_test,y_test)

##plotting confusion matrix
y_predicted=model1.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')



