# Project: Pima Diabetes Prediction
# Dataset Location: JBrownLee Repository on Github
# Link to Dataset: "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

# Created By Aaditya Raval on 08/27/2021

import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)

# Create dataframe
col_name = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.DataFrame(data, columns=col_name)
print(df.dtypes)

# Create train and test split
TrainSet = df.drop(columns='class')
TestSet = df['class']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(TrainSet, TestSet, random_state=21, test_size=0.2)

# Standard Scaler for data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_standard = scaler.transform(X_train)
X_test_standard = scaler.transform(X_test)

# SVM and KNN models
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('SVM', SVC(gamma='auto')))

# Cross validation
training_results = []
prediction_results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train_standard, Y_train, cv=kfold, 
                                 scoring='accuracy')
    cv_predictions = cross_val_score(model, X_test_standard, Y_test, cv=kfold, 
                                 scoring='accuracy')
    training_results.append(cv_results)
    prediction_results.append(cv_predictions)
    names.append(name)
    print('%s training: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Cross-validation prediction
print('KNN prediction: %.2f' % (prediction_results[0].mean()))
print('SVM prediction: %.2f' % (prediction_results[1].mean()))

# Model prediction
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    
    # Evaluation matrics
    print("Confusion Matrix: ")
    print(confusion_matrix(Y_test, predictions))
