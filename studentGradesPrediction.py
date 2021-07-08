import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# This is a very simple linear regression model to predict student grades in math class based on a few features
data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "study time", "failures", "absences"]]
predict = "G3"
X = np.array(data.drop(["G1", "G2", "G3"], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Coefficients: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
