#Importing required libraries
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd 

#Creating a data frame with pandas
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

#Could get away with dropping samples with missing values, however it is better practice to make them outliers.
#df.dropna('?')
df.replace('?', -99999, inplace=True)

#ID is arbitrary, does not contribute towards prediction
df.drop(['id'], 1, inplace=True)

#Create features and class matrices to train the algorithm on
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

#Train a model
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.25)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

#Testing accuracy of this model
accuracy = clf.score(X_test, y_test)
print(accuracy)
#Note: accuracy ranges from roughly 0.94 to 0.96. This is because 'k' may be different everytime the model is run.

#Testing unseen data i.e. new patient
#new_patient = np.array([3,2,5,1,1,2,10,8,4])
#diagnosis = clf.predict(new_patient)
#print(diagnosis)