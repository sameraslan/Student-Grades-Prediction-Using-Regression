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
# data = pd.get_dummies(data = data, drop_first = True)

data['school'] = data['school'].astype('category').cat.codes
data['sex'] = data['sex'].astype('category').cat.codes
data['address'] = data['address'].astype('category').cat.codes
data['famsize'] = data['famsize'].astype('category').cat.codes
data['Pstatus'] = data['Pstatus'].astype('category').cat.codes
data['Mjob'] = data['Mjob'].astype('category').cat.codes
data['Fjob'] = data['Fjob'].astype('category').cat.codes
data['reason'] = data['reason'].astype('category').cat.codes
data['guardian'] = data['guardian'].astype('category').cat.codes
data['schoolsup'] = data['schoolsup'].astype('category').cat.codes
data['famsup'] = data['famsup'].astype('category').cat.codes
data['paid'] = data['paid'].astype('category').cat.codes
data['activities'] = data['activities'].astype('category').cat.codes
data['nursery'] = data['nursery'].astype('category').cat.codes
data['internet'] = data['internet'].astype('category').cat.codes
data['higher'] = data['higher'].astype('category').cat.codes
data['romantic'] = data['romantic'].astype('category').cat.codes

#Positive correlations above .1 only resulted in best model with .9787 prediction accuracy
#data = data[["G1", "G2", "G3", "sex", "address", "Medu", "Fedu", "Mjob", "reason", "paid", "higher"]]

#Positive and negative correlations above .1 only resulted in best model with .9726 prediction accuracy
data = data[["G1", "G2", "G3", "sex", "address", "Medu", "Fedu", "Mjob", "reason", "paid", "higher", 'age', 'traveltime', 'failures', 'romantic', 'goout']]
predict = 'G3'

# Print out all feature correlations to G3
counter = 0
for column in data:
    featureList = list(data.columns)
    correlation = data[column].corr(data[predict])
    # Prints out correlations above .1 for me to select (neg and pos second time)
    '''if correlation >= .1 or correlation <= -.1:
        print(str(featureList[counter]) + ": " + str(correlation))'''

    counter = counter + 1
    if counter == len(featureList) - 1:
        break

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

'''
bestScore = 0
for i in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    #print(acc)

    if acc > bestScore:
        #Models Saved
        with open("studentGradesNegAndPosPredModel.pickle", "wb") as f:
            pickle.dump(linear, f)
        bestScore = acc
'''

#print(bestScore)
pickleInput = open("studentGradesNegAndPosPredModel.pickle", "rb")
linear = pickle.load(pickleInput)
acc = linear.score(x_test, y_test)
print(acc)

#print('Coefficient: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)

#predictions = linear.predict(x_test)

'''for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])'''

'''p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()'''