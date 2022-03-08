import numpy as np
import pandas as pd 
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# This is the data
data = pd.read_csv('risk.csv')

# For the x-axis, I choose age, income, gender 
x = data [[
    'INCOMEB',
    'AGEB',
    'MORTGAGEB',
]]
y = data[['RISKB']]

# Creating model
knn = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform')

# Testing and prediction
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("predictions: ", prediction)
print("accuracy: ", accuracy)