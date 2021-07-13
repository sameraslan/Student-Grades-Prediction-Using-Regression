# This is a very simple linear regression model to predict student grades in math class based on a few features
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
# data = data[["G1", "G2", "G3", "studytime", "failures", "Mjob", "Fjob", "absences"]]
data = pd.get_dummies(data = data, drop_first = True)


predict = "G3"



X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

"""linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

with open("studentGradesModel.pickle", "wb") as f:
    pickle.dump(linear, f)"""

pickleInput = open("studentGradesModel.pickle", "rb")
linear = pickle.load(pickleInput)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
