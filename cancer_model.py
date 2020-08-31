# Cancer

# Importing libraries
import numpy as np
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('cancer.csv')
dataset = dataset.iloc[:,:-1]
X = dataset.iloc[:,2:].values
y = dataset.iloc[:,1].values

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting SVC to the dataset
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Pickling the scaler
import pickle
file = open('cancer_scaler.pkl','wb')
pickle.dump(sc_X,file)

# Pickling the model
file = open('cancer.pkl','wb')
pickle.dump(classifier,file)









