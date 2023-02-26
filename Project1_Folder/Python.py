import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

DataSet = pd.read_csv(r"/Users/zakariaghazal/Desktop/ACME-HappinessSurvey2020.csv")

X = DataSet.iloc[:,1:].values

#print("X data is ", X)

Y = DataSet.iloc[:,0].values

#print("Y data is ", Y)


#Using KNeighbors model 
'''
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25 , random_state =1)

#print(" X train dataset is : ", X_train)

#print(" X test dataset is : ", X_test)

#print(" Y train dataset is : ", Y_train)

#print(" Y test dataset is : ", Y_test)

from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p=2)
Classifier.fit(X_train, Y_train)
Y_Pred = Classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
hm = confusion_matrix(Y_test, Y_Pred)
print(classification_report(Y_test, Y_Pred))
'''

#Using DecisionTree model
'''
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3 , random_state =3)

HappinesTree = DecisionTreeClassifier(criterion="entropy", max_depth = 2)
HappinesTree.fit(X_train,Y_train)
Y_Pred = HappinesTree.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
hm = confusion_matrix(Y_test, Y_Pred)
print(classification_report(Y_test, Y_Pred))
'''


'''
as we are reviewing the dataset and the features(attributes) that we have used in our algorithm model 
we have noticed that some of the features are not really necessary to consider it in our training 
features so we decided to remove the below attributes from the dataset:
    X3 = I ordered everything I wanted to order
    X4 = I paid a good price for my order
    X6 = the app makes ordering easy for me
    
please find below the model after deleteing this 3 features
    
'''

DataSetModified = DataSet.drop(['X3', 'X4', 'X6'], axis=1)

#print(DataSetModified)


X1 = DataSetModified.iloc[:,1:].values

#print("X1 data is ", X1)

Y1 = DataSetModified.iloc[:,0].values

#print("Y1 data is ", Y1)



#Using KNeighbors model after modification
'''
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size = 0.25 , random_state =1)

print(" X1 train dataset is : ", X_train)

print(" X1 test dataset is : ", X_test)

print(" Y1 train dataset is : ", Y_train)

print(" Y1 test dataset is : ", Y_test)

from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p=2)
Classifier.fit(X_train, Y_train)
Y_Pred = Classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
hm = confusion_matrix(Y_test, Y_Pred)
print(classification_report(Y_test, Y_Pred))
'''



#Using DecisionTree model after modification
'''
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size = 0.3 , random_state =3)

HappinesTree = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
HappinesTree.fit(X_train,Y_train)
Y_Pred = HappinesTree.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
hm = confusion_matrix(Y_test, Y_Pred)
print(classification_report(Y_test, Y_Pred))
'''




